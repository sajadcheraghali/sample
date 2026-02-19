# llm.py
from openai import OpenAI
from config import OPENROUTER_API_KEY, BASE_URL

client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url=BASE_URL
)

def call_llm(prompt):
    response = client.chat.completions.create(
        model="deepseek/deepseek-chat",  # مدل ارزان و مناسب
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )

    return response.choices[0].message.content