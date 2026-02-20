import time
import uuid
from embedding import get_embedding
from llm import call_llm
import chromadb

class STM:
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.buffer = []
        self.client = chromadb.Client()
        self.collection = self.client.create_collection("stm_memory")

    # اضافه کردن مکالمه
    def add_message(self, role, content):
        message = {
            "id": str(uuid.uuid4()),
            "role": role,
            "content": content,
            "timestamp": time.time()
        }

        self.buffer.append(message)

        # embedding و ذخیره در vector store
        embedding = get_embedding(content)

        self.collection.add(
            documents=[content],
            embeddings=[embedding],
            ids=[message["id"]],
            metadatas=[{"timestamp": message["timestamp"]}]
        )

        # اگر از window بیشتر شد → خلاصه کن
        if len(self.buffer) > self.window_size:
            self._summarize_old_messages()

    # خلاصه‌سازی مکالمات قدیمی
    def _summarize_old_messages(self):
        texts = [msg["content"] for msg in self.buffer[:-1]]

        combined_text = "\n".join(texts)

        summary_prompt = f"""
        Summarize the following conversation briefly:
        {combined_text}
        """

        summary = call_llm(summary_prompt)

        # فقط summary + آخرین پیام را نگه می‌داریم
        self.buffer = [{
            "id": str(uuid.uuid4()),
            "role": "system",
            "content": summary,
            "timestamp": time.time()
        }] + [self.buffer[-1]]

    # بازیابی top-k مکالمات مرتبط
    def retrieve(self, query, k=3):
        query_embedding = get_embedding(query)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k
        )

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]

        # مرتب‌سازی بر اساس timestamp
        sorted_results = sorted(
            zip(documents, metadatas),
            key=lambda x: x[1]["timestamp"]
        )

        return [doc for doc, _ in sorted_results]