from memory.ltm import LTM
import time

ltm = LTM()

# شبیه‌سازی سشن اول
session1 = [
    {"id": "1", "content": "I like sushi.", "timestamp": time.time()},
    {"id": "2", "content": "Japanese food is great.", "timestamp": time.time()}
]

ltm.add_session(session1)

# شبیه‌سازی سشن دوم
session2 = [
    {"id": "3", "content": "I prefer Chinese car.", "timestamp": time.time()}
]

ltm.add_session(session2)



# تست retrieval
results = ltm.retrieve_groups("What food do I like?")
print("Relevant Groups:")
for r in results:
    print("-", r)