# memory/ltm.py

import uuid
import time
import re
from embedding import get_embedding
from llm import call_llm
import chromadb
import json
from retrieval.iterative import decide_action


class LTM:
    def __init__(self):
        self.client = chromadb.Client()
        self.group_collection = self.client.create_collection("ltm_groups")
        self.groups = {}
        self.topic_to_id = {}

    def _sanitize_topic_id(self, topic):
        """تبدیل topic به فرمت معتبر برای نام collection"""
        # تبدیل فاصله و کاراکترهای خاص به underscore
        sanitized = re.sub(r'[^a-zA-Z0-9._-]', '_', topic)
        
        # حذف underscore، نقطه و dash از ابتدا و انتها
        sanitized = sanitized.strip('_.-')
        
        # جایگزینی نقطه و dash میانی با underscore (برای اطمینان)
        sanitized = re.sub(r'[.-]', '_', sanitized)
        
        # اگر خالی شد یا کوتاه بود، یک id پیش‌فرض بساز
        if len(sanitized) < 3:
            sanitized = f"topic_{uuid.uuid4().hex[:8]}"
        
        # محدود کردن طول
        if len(sanitized) > 512:
            sanitized = sanitized[:512]
        
        # اطمینان از شروع و پایان با حرف یا عدد
        sanitized = sanitized.strip('_.-')
        
        # بررسی نهایی
        if len(sanitized) < 3 or not re.match(r'^[a-zA-Z0-9]', sanitized) or not re.search(r'[a-zA-Z0-9]$', sanitized):
            sanitized = f"topic_{uuid.uuid4().hex[:8]}"
        
        return sanitized

    def add_session(self, conversations):
        combined_text = "\n".join([c["content"] for c in conversations])

        topic_prompt = f"""
        Identify the main topic of the following conversation in 3 words max:

        {combined_text}
        """

        topic = call_llm(topic_prompt).strip()
        topic_id = self._sanitize_topic_id(topic)

        if topic_id not in self.groups:
            self._create_new_group(topic, topic_id, conversations)
        else:
            self._add_to_existing_group(topic, topic_id, conversations)

    def _create_new_group(self, topic, topic_id, conversations):
        summary = self._generate_summary(conversations)
        summary_embedding = get_embedding(summary)

        # ساخت collection با نام sanitized
        conv_collection = self.client.create_collection(f"group_{topic_id}")

        for conv in conversations:
            embedding = get_embedding(conv["content"])
            conv_collection.add(
                documents=[conv["content"]],
                embeddings=[embedding],
                ids=[conv["id"]],
                metadatas=[{"timestamp": conv["timestamp"]}]
            )

        self.groups[topic_id] = {
            "topic": topic,
            "summary": summary,
            "summary_embedding": summary_embedding,
            "collection": conv_collection
        }

        self.topic_to_id[topic] = topic_id

        self.group_collection.add(
            documents=[summary],
            embeddings=[summary_embedding],
            ids=[topic_id],
            metadatas=[{"topic": topic}]
        )

    def _add_to_existing_group(self, topic, topic_id, conversations):
        conv_collection = self.groups[topic_id]["collection"]

        for conv in conversations:
            embedding = get_embedding(conv["content"])
            conv_collection.add(
                documents=[conv["content"]],
                embeddings=[embedding],
                ids=[conv["id"]],
                metadatas=[{"timestamp": conv["timestamp"]}]
            )

        updated_summary = self._generate_summary(
            self._get_all_conversations(topic_id)
        )

        updated_embedding = get_embedding(updated_summary)

        self.groups[topic_id]["summary"] = updated_summary
        self.groups[topic_id]["summary_embedding"] = updated_embedding

        self.group_collection.upsert(
            documents=[updated_summary],
            embeddings=[updated_embedding],
            ids=[topic_id],
            metadatas=[{"topic": topic}]
        )

    def _generate_summary(self, conversations):
        combined_text = "\n".join([c["content"] for c in conversations])

        summary_prompt = f"""
        Summarize the core theme of these conversations:

        {combined_text}
        """

        return call_llm(summary_prompt)

    def _get_all_conversations(self, topic_id):
        """بازیابی تمام مکالمات یک گروه از collection"""
        conv_collection = self.groups[topic_id]["collection"]
        results = conv_collection.get()
        
        conversations = []
        if results["documents"]:
            for i, doc in enumerate(results["documents"]):
                conversations.append({
                    "content": doc,
                    "id": results["ids"][i],
                    "timestamp": results["metadatas"][i]["timestamp"] if results["metadatas"] else 0
                })
        
        return conversations

    def retrieve_ltm_group(self, query, k=2, r=3):
        """
        LTM Group Retrieval
        """
        query_embedding = get_embedding(query)

        group_results = self.group_collection.query(
            query_embeddings=[query_embedding],
            n_results=k
        )

        top_groups = group_results["ids"][0]

        if not top_groups:
            return []

        best_group = top_groups[0]
        conv_collection = self.groups[best_group]["collection"]

        conv_results = conv_collection.query(
            query_embeddings=[query_embedding],
            n_results=r
        )

        documents = conv_results["documents"][0]
        metadatas = conv_results["metadatas"][0]

        sorted_results = sorted(
            zip(documents, metadatas),
            key=lambda x: x[1]["timestamp"]
        )

        return [doc for doc, _ in sorted_results]

    def get_topic_name(self, topic_id):
        """دریافت نام اصلی topic از topic_id"""
        if topic_id in self.groups:
            return self.groups[topic_id].get("topic", topic_id)
        return topic_id

    def retrieve_ltm_iterative(self, query, k=2, r=3, max_rounds=3):
        rounds = 0
        visited_groups = set()
        search_history = []
        final_results = []
        current_query = query

        while rounds < max_rounds:
            query_embedding = get_embedding(current_query)

            group_results = self.group_collection.query(
                query_embeddings=[query_embedding],
                n_results=k
            )

            top_groups = group_results["ids"][0]

            if not top_groups:
                break

            for group in top_groups:
                if group in visited_groups:
                    continue

                visited_groups.add(group)

                conv_collection = self.groups[group]["collection"]

                conv_results = conv_collection.query(
                    query_embeddings=[query_embedding],
                    n_results=r
                )

                conversations = conv_results["documents"][0]
                group_summary = self.groups[group]["summary"]

                decision_raw = decide_action(
                    current_query,
                    group_summary,
                    conversations,
                    search_history
                )

                try:
                    decision = json.loads(decision_raw)
                except:
                    break

                action = decision["action"]

                if action == "END":
                    final_results = conversations
                    return final_results

                elif action == "JUMP":
                    continue

                elif action == "REWRITE":
                    current_query = decision["new_query"]
                    search_history.append(current_query)
                    visited_groups = set()
                    break

            rounds += 1

        return final_results