from memory.stm import STM

stm = STM(window_size=3)

# شبیه‌سازی مکالمه
stm.add_message("user", "I like sushi.")
stm.add_message("assistant", "Great choice!")
stm.add_message("user", "I also enjoy ramen.")
stm.add_message("assistant", "Japanese food is delicious.")

# این پیام باعث خلاصه‌سازی می‌شود
stm.add_message("user", "What food do I prefer?")

# تست retrieval
results = stm.retrieve("What food do I like?")

print("Retrieved:")
for r in results:
    print("-", r)