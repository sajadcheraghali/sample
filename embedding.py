# embedding.py
from openai import OpenAI
from config import OPENROUTER_API_KEY, BASE_URL

client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url=BASE_URL
)

def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )

    return response.data[0].embedding