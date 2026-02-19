# vector_store.py
import chromadb

client = chromadb.Client()
collection = client.create_collection("test_memory")

def add_memory(text, embedding):
    collection.add(
        documents=[text],
        embeddings=[embedding],
        ids=[str(hash(text))]
    )

def search_memory(query_embedding):
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=2
    )
    return results