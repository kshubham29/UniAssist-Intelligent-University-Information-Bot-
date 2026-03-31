import json
import os
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

# Load scraped data
with open("data/ncu_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

texts = [doc["content"] for doc in data]

print(f"Loaded {len(texts)} pages")

# Chunk manually
chunk_size = 1000
overlap = 200

chunks = []

for text in texts:
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i + chunk_size]
        chunks.append(chunk)

print(f"Total chunks created: {len(chunks)}")

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Create embeddings
embeddings = model.encode(chunks, show_progress_bar=True)

# Save locally
os.makedirs("vector_db", exist_ok=True)

with open("vector_db/chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)

with open("vector_db/embeddings.npy", "wb") as f:
    np.save(f, embeddings)

print("✅ Vector database built successfully.")