from memory.ltm import LTM
import time
import uuid

ltm = LTM()

session1 = [
    {"id": str(uuid.uuid4()), "content": "I like sushi.", "timestamp": time.time()},
    {"id": str(uuid.uuid4()), "content": "Japanese food is my favorite.", "timestamp": time.time()}
]

ltm.add_session(session1)

session2 = [
    {"id": str(uuid.uuid4()), "content": "I like play football.", "timestamp": time.time()},
    {"id": str(uuid.uuid4()), "content": "Exercising is good for your health.", "timestamp": time.time()}
]

ltm.add_session(session2)

query = "What food do I prefer?"

results = ltm.retrieve_ltm_group(query)

print("LTM Group Retrieved:")
for r in results:
    print("-", r)