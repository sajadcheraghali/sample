# memory/ltm.py

import uuid
import time
import re
from embedding import get_embedding
from llm import call_llm
import chromadb


class LTM:
    def __init__(self):
        self.client = chromadb.Client()
        self.group_collection = self.client.create_collection("ltm_groups")
        self.groups = {}
        self.topic_to_id = {}  # نگاشت topic اصلی به id sanitized

    def _sanitize_topic_id(self, topic):
        """تبدیل topic به فرمت معتبر برای نام collection"""
        # تبدیل فاصله و کاراکترهای خاص به underscore
        sanitized = re.sub(r'[^a-zA-Z0-9._-]', '_', topic)
        # حذف underscore از ابتدا و انتها
        sanitized = sanitized.strip('_')
        # اگر خالی شد یا کوتاه بود، یک id پیش‌فرض بساز
        if len(sanitized) < 3:
            sanitized = f"topic_{uuid.uuid4().hex[:8]}"
        # محدود کردن طول
        if len(sanitized) > 512:
            sanitized = sanitized[:512]
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
            "topic": topic,  # ذخیره topic اصلی برای نمایش
            "summary": summary,
            "summary_embedding": summary_embedding,
            "collection": conv_collection
        }

        self.topic_to_id[topic] = topic_id  # نگاشت topic به id

        self.group_collection.add(
            documents=[summary],
            embeddings=[summary_embedding],
            ids=[topic_id],
            metadatas=[{"topic": topic}]  # ذخیره topic اصلی در metadata
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

        # 1️⃣ انتخاب top-k گروه
        group_results = self.group_collection.query(
            query_embeddings=[query_embedding],
            n_results=k
        )

        top_groups = group_results["ids"][0]

        if not top_groups:
            return []

        # 2️⃣ از بهترین گروه top-r مکالمه
        best_group = top_groups[0]
        conv_collection = self.groups[best_group]["collection"]

        conv_results = conv_collection.query(
            query_embeddings=[query_embedding],
            n_results=r
        )

        documents = conv_results["documents"][0]
        metadatas = conv_results["metadatas"][0]

        # 3️⃣ مرتب‌سازی بر اساس timestamp
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