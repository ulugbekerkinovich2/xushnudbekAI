from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
import os
from dotenv import load_dotenv
import pickle
import numpy as np
from numpy.linalg import norm
import json
from fastapi.middleware.cors import CORSMiddleware
# Load environment variables
load_dotenv()
client = OpenAI(api_key=os.getenv("API_KEY"))

# FastAPI app
app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # yoki ["http://example.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Fayl nomlari
EMBEDDING_FILE = "talim_embeddings.pkl"

# config.json ichidan assistant_id va vector_store_id ni o‘qiymiz
with open("config.json", "r") as f:
    config = json.load(f)

ASSISTANT_ID = config.get("assistant_id")
VECTOR_STORE_ID = config.get("vector_store_id")

# So‘rov uchun pydantic model
class Query(BaseModel):
    question: str

# Matnni embeddingga aylantirish
def get_embedding(text: str) -> list:
    response = client.embeddings.create(
        input=[text],
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

# Embedding faylni yuklash
def load_embeddings():
    if os.path.exists(EMBEDDING_FILE):
        with open(EMBEDDING_FILE, "rb") as f:
            return pickle.load(f)
    raise FileNotFoundError("Embedding fayli topilmadi. Avval uni yaratish kerak.")

# Eng mos keladigan qatorni topish
# def find_best_matching_line(lines: list, embeddings: list, query: str):
#     query_emb = get_embedding(query)
#     similarities = [
#         np.dot(query_emb, emb) / (norm(query_emb) * norm(emb))
#         for emb in embeddings
#     ]
#     best_index = int(np.argmax(similarities))
#     return lines[best_index], similarities[best_index]
def find_best_matching_line(lines: list, embeddings: list, query: str, top_k: int = 3):
    query_emb = get_embedding(query)
    similarities = [
        np.dot(query_emb, emb) / (norm(query_emb) * norm(emb))
        for emb in embeddings
    ]
    top_indices = np.argsort(similarities)[-top_k:][::-1]  # top k indexlar
    top_lines = [lines[i] for i in top_indices]
    top_scores = [similarities[i] for i in top_indices]
    return list(zip(top_lines, top_scores))

def find_best_matching_line(lines: list, embeddings: list, query: str, top_k: int = 3):
    query_emb = get_embedding(query)
    similarities = [
        np.dot(query_emb, emb) / (norm(query_emb) * norm(emb))
        for emb in embeddings
    ]
    top_indices = np.argsort(similarities)[-top_k:][::-1]  # yuqori k indexlar
    top_lines = [lines[i] for i in top_indices]
    top_scores = [similarities[i] for i in top_indices]
    return list(zip(top_lines, top_scores))

@app.post("/ask")
def ask_question(query: Query):
    try:
        lines, embeddings = load_embeddings()
    except Exception as e:
        return {"error": str(e)}

    top_matches = find_best_matching_line(lines, embeddings, query.question, top_k=3)
    
    # Assistantga savol beramiz
    thread = client.beta.threads.create()
    client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=query.question
    )
    run = client.beta.threads.runs.create_and_poll(
        thread_id=thread.id,
        assistant_id=ASSISTANT_ID
    )
    messages = client.beta.threads.messages.list(thread_id=thread.id)
    answer = messages.data[0].content[0].text.value

    # Sourcelarni matn qilib formatlaymiz
    formatted_sources = "\n\n".join(
        [f"{i+1}. {line.strip()} (score: {score:.3f})" for i, (line, score) in enumerate(top_matches)]
    )

    return {
        "answer": answer,
        "source": formatted_sources
    }

# /ask endpoint
# @app.post("/ask")
# def ask_question(query: Query):
#     try:
#         lines, embeddings = load_embeddings()
#     except Exception as e:
#         return {"error": str(e)}

#     best_line, score = find_best_matching_line(lines, embeddings, query.question)

#     if not ASSISTANT_ID:
#         return {"error": "❗ assistant_id config.json faylida topilmadi."}

#     # GPT assistantga so‘rov yuborish
#     try:
#         thread = client.beta.threads.create()

#         client.beta.threads.messages.create(
#             thread_id=thread.id,
#             role="user",
#             content=query.question
#         )

#         run = client.beta.threads.runs.create_and_poll(
#             thread_id=thread.id,
#             assistant_id=ASSISTANT_ID
#         )

#         messages = client.beta.threads.messages.list(thread_id=thread.id)
#         print(messages)
#         answer = messages.data[0].content[0].text.value

#         return {
#             "answer": answer,
#             "source": best_line
#         }

#     except Exception as e:
#         return {"error": f"Assistant so‘rovda xato: {str(e)}"}
