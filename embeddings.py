from sentence_transformers import SentenceTransformer
from chunking import chunks

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings_list = []
for file, content in chunks:
    embeddings = model.encode(content)
    embeddings_list.append((file, content, embeddings))
    print(f"Embeddings for {file}:")
    print(embeddings)
    print("-" * 40)
print(len(embeddings_list))
