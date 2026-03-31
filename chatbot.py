import os
import pickle
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai


# ==========================================
# 🔐 LOAD ENV VARIABLES
# ==========================================
load_dotenv()
api_key = "AIzaSyDvu0dozL4pOqfS9QO7alR70QkVqX5kceY"
print("API KEY BEING USED:", api_key)

if not api_key:
    raise ValueError(
        "GOOGLE_API_KEY not found.\n"
        "Make sure your .env file contains:\n"
        "GOOGLE_API_KEY=your_key_here"
    )

genai.configure(api_key=api_key)

# Use latest stable Gemini model
model = genai.GenerativeModel("gemini-2.5-flash")

print("✅ Gemini configured successfully.")


# ==========================================
# 📦 LOAD VECTOR DATABASE
# ==========================================
print("📦 Loading vector database...")

try:
    with open("vector_db/chunks.pkl", "rb") as f:
        chunks = pickle.load(f)

    embeddings = np.load("vector_db/embeddings.npy")

except FileNotFoundError:
    raise FileNotFoundError(
        "Vector database not found.\n"
        "Run: python build_vectorstore.py"
    )

embed_model = SentenceTransformer("all-MiniLM-L6-v2")

print("✅ Vector DB loaded successfully.\n")


# ==========================================
# 🔎 RETRIEVAL FUNCTION
# ==========================================
def retrieve_context(query, top_k=50):
    query_embedding = embed_model.encode([query])
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    selected_chunks = [chunks[i] for i in top_indices]
    context = "\n\n".join(selected_chunks)

    # Safety trim (Gemini input limit)
    return context[:30000]


# ==========================================
# 💬 ASK FUNCTION
# ==========================================
def ask(query):
    context = retrieve_context(query)

    prompt = f"""
You are UniAssist NCU, the official AI assistant of The NorthCap University.

INSTRUCTIONS:
- Answer strictly using provided context.
- Do NOT hallucinate.
- Structure answer clearly using headings and bullet points.
- If answer not found in context, say:
  "The information is not available in the official university data."

Context:
{context}

Question:
{query}
"""

    try:
        response = model.generate_content(prompt)
        return response.text

    except Exception as e:
        return f"❌ Error generating response:\n{str(e)}"


# ==========================================
# 🖥 CLI LOOP
# ==========================================
if __name__ == "__main__":
    print("=" * 60)
    print("🎓 NCU UniAssist Chatbot Ready")
    print("Type 'exit' to stop.")
    print("=" * 60)

    while True:
        question = input("\nAsk: ")

        if question.lower() == "exit":
            print("👋 Exiting UniAssist...")
            break

        print("\nAnswer:\n")
        answer = ask(question)
        print(answer)
        print("\n" + "-" * 60)