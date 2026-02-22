from memory.ltm import LTM
import uuid
import time

ltm = LTM()

session1 = [
    {"id": str(uuid.uuid4()), "content": "I love sushi.", "timestamp": time.time()},
    {"id": str(uuid.uuid4()), "content": "Japanese cuisine is my favorite.", "timestamp": time.time()}
]

session2 = [
    {"id": str(uuid.uuid4()), "content": "I dislike burgers.", "timestamp": time.time()}
]

ltm.add_session(session1)
ltm.add_session(session2)

query = "What should I eat for lunch?"

results = ltm.retrieve_ltm_iterative(query)

print("Iterative Retrieval:")
for r in results:
    print("-", r)