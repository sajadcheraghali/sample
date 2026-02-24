# test_vector.py
from embedding import get_embedding
from vector_store import add_memory, search_memory

text1 = "I like sushi."
text2 = "I prefer Chinese food."
text3 = "i want to go to park"

add_memory(text1, get_embedding(text1))
add_memory(text2, get_embedding(text2))
add_memory(text3, get_embedding(text3))

query = "What food do I like?"
results = search_memory(get_embedding(query))

print(results)