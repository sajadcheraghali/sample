from memory.stm import STM
from memory.ltm import LTM
from retrieval.hybrid import HybridRetriever
import uuid
import time

# ساخت STM و LTM
stm = STM(window_size=3)
ltm = LTM()

# شبیه سازی سشن قبلی (می رود به LTM)
session1 = [
    {"id": str(uuid.uuid4()), "content": "I love sushi.", "timestamp": time.time()},
    {"id": str(uuid.uuid4()), "content": "Japanese food is my favorite.", "timestamp": time.time()}
]

ltm.add_session(session1)

# سشن فعلی در STM
stm.add_message("user", "I am thinking about lunch.")
stm.add_message("assistant", "What are you considering?")
stm.add_message("user", "Maybe something Asian.")

# ساخت hybrid
hybrid = HybridRetriever(stm, ltm)

# پرسش
query = "What should I eat?"

response = hybrid.retrieve_and_generate(query, debug=True)

print("\nFINAL RESPONSE:")
print(response)