from openai import OpenAI
import numpy as np
from numpy.linalg import norm
from dotenv import load_dotenv
from tqdm import tqdm
import os
import pickle

# .env dan yuklash
load_dotenv()
client = OpenAI(api_key=os.getenv("API_KEY"))

EMBEDDING_FILE = "talim_embeddings.pkl"
TEXT_FILE = "talim.txt"

def get_embedding(text: str) -> list:
    response = client.embeddings.create(
        input=[text],
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

def embed_all_lines(filename: str):
    if os.path.exists(EMBEDDING_FILE):
        print("📂 Oldingi embeddinglar topildi. Yuklanmoqda...")
        with open(EMBEDDING_FILE, "rb") as f:
            return pickle.load(f)

    print("⏳ Embeddinglar hisoblanmoqda...")
    with open(filename, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    embeddings = []
    for line in tqdm(lines, desc="🔁 Satrlarni embedding qilish"):
        emb = get_embedding(line)
        embeddings.append(emb)

    # Natijani saqlash
    with open(EMBEDDING_FILE, "wb") as f:
        pickle.dump((lines, embeddings), f)

    return lines, embeddings

def find_best_matching_line(lines: list, embeddings: list, query: str):
    query_emb = get_embedding(query)
    similarities = [
        np.dot(query_emb, emb) / (norm(query_emb) * norm(emb))
        for emb in embeddings
    ]
    best_index = int(np.argmax(similarities))
    return lines[best_index], similarities[best_index]

# if __name__ == "__main__":
#     query = "4:0"

#     lines, embeddings = embed_all_lines(TEXT_FILE)

#     print("🔍 Eng mos satr aniqlanmoqda...")
#     best_line, score = find_best_matching_line(lines, embeddings, query)

#     print(f"\n✅ Eng mos satr (score={score:.4f}):\n{best_line}")
