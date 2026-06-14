from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
embedding = model.encode("Hello world")

print(len(embedding))  # Output: 384
print(type(embedding))  # Output: <class 'numpy.ndarray'>
print(embedding)  # Output: [0.123, -0.456, ..., 0.567]