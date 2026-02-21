import uuid
import time
from embedding import get_embedding
from llm import call_llm
import chromadb


class LTM:
    def __init__(self):
        self.client = chromadb.Client()
        self.group_collection = self.client.create_collection("ltm_groups")
        self.groups = {}  # in-memory structure

    # انتقال یک سشن کامل به LTM
    def add_session(self, conversations):
        """
        conversations: list of dicts from STM buffer
        """

        combined_text = "\n".join([c["content"] for c in conversations])

        # 1️⃣ از LLM بخواه topic را تشخیص دهد
        topic_prompt = f"""
        Identify the main topic of the following conversation in 3 words max:

        {combined_text}
        """

        topic = call_llm(topic_prompt).strip()

        # اگر group وجود ندارد → بساز
        if topic not in self.groups:
            self._create_new_group(topic, conversations)
        else:
            self._add_to_existing_group(topic, conversations)

    # ساخت گروه جدید
    def _create_new_group(self, topic, conversations):
        summary = self._generate_summary(conversations)
        summary_embedding = get_embedding(summary)

        self.groups[topic] = {
            "group_id": str(uuid.uuid4()),
            "summary": summary,
            "summary_embedding": summary_embedding,
            "conversations": conversations
        }

        # ذخیره summary در vector DB برای retrieval
        self.group_collection.add(
            documents=[summary],
            embeddings=[summary_embedding],
            ids=[topic]
        )

    # اضافه کردن به گروه موجود
    def _add_to_existing_group(self, topic, conversations):
        self.groups[topic]["conversations"].extend(conversations)

        # آپدیت summary
        updated_summary = self._generate_summary(
            self.groups[topic]["conversations"]
        )

        updated_embedding = get_embedding(updated_summary)

        self.groups[topic]["summary"] = updated_summary
        self.groups[topic]["summary_embedding"] = updated_embedding

        # update vector store
        self.group_collection.upsert(
            documents=[updated_summary],
            embeddings=[updated_embedding],
            ids=[topic]
        )

    # تولید summary برای گروه
    def _generate_summary(self, conversations):
        combined_text = "\n".join([c["content"] for c in conversations])

        summary_prompt = f"""
        Summarize the core theme of these conversations:

        {combined_text}
        """

        return call_llm(summary_prompt)

    # بازیابی top-k گروه مرتبط
    def retrieve_groups(self, query, k=2):
        query_embedding = get_embedding(query)

        results = self.group_collection.query(
            query_embeddings=[query_embedding],
            n_results=k
        )

        return results["documents"][0]