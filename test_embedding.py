# test_embedding.py
from embedding import get_embedding

vec = get_embedding("This is a memory test.")
print("Vector length:", len(vec))
print(vec[:5])