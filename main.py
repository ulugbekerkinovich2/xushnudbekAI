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
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi import Request


limiter = Limiter(key_func=get_remote_address)


app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)





load_dotenv()
client = OpenAI(api_key=os.getenv("API_KEY"))




app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


EMBEDDING_FILE = "talim_embeddings.pkl"


with open("config.json", "r") as f:
    config = json.load(f)

ASSISTANT_ID = config.get("assistant_id")
VECTOR_STORE_ID = config.get("vector_store_id")


class Query(BaseModel):
    question: str
@app.get("/")
def home():
    return {"message": "server is running"}

def get_embedding(text: str) -> list:
    response = client.embeddings.create(
        input=[text],
        model="text-embedding-3-small"
    )
    return response.data[0].embedding


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
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    top_lines = [lines[i] for i in top_indices]
    top_scores = [similarities[i] for i in top_indices]
    return list(zip(top_lines, top_scores))

# def find_best_matching_line(entries: list, embeddings: list, query: str, top_k: int = 3):
#     query_emb = get_embedding(query)
#     similarities = [
#         np.dot(query_emb, emb) / (norm(query_emb) * norm(emb))
#         for emb in embeddings
#     ]
#     top_indices = np.argsort(similarities)[-top_k:][::-1]
#     top_results = []
#     for i in top_indices:
#         item = entries[i]
#         top_results.append({
#             "matn": item["matn"],
#             "manba": item.get("manba"),
#             "link": item.get("link"),
#             "score": similarities[i]
#         })
#     return top_results


# def find_best_matching_line(lines: list, embeddings: list, query: str, top_k: int = 3):
#     query_emb = get_embedding(query)
#     similarities = [
#         np.dot(query_emb, emb) / (norm(query_emb) * norm(emb))
#         for emb in embeddings
#     ]
#     top_indices = np.argsort(similarities)[-top_k:][::-1]
#     top_lines = [lines[i] for i in top_indices]
#     top_scores = [similarities[i] for i in top_indices]
#     return list(zip(top_lines, top_scores))
import re

def extract_links_from_answer(answer: str):
    """Answer ichidan [text](link) formatidagi havolalarni ajratib oladi."""
    link_pattern = r'\[([^\]]+)\]\((https?://[^\)]+)\)'
    matches = re.findall(link_pattern, answer)
    
    links = [{"title": title, "url": url} for title, url in matches]
    # Clean answer text (remove markdown links)
    cleaned_answer = re.sub(link_pattern, r'\1', answer)

    return cleaned_answer.strip(), links

@limiter.limit("5/minute")
@app.post("/ask")
def ask_question(query: Query, request: Request):
    try:
        lines, embeddings = load_embeddings()
    except Exception as e:
        return {"error": str(e)}

    top_matches = find_best_matching_line(lines, embeddings, query.question, top_k=3)
    

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
    raw_answer = messages.data[0].content[0].text.value.strip()

    # [title](url) formatdagi linklarni ajratib olish
    cleaned_answer, links = extract_links_from_answer(raw_answer)

    formatted_sources = "\n\n".join(
        [f"{i+1}. {line.strip()} (score: {score:.3f})" for i, (line, score) in enumerate(top_matches)]
    )

    return {
        "answer": {
            "text": cleaned_answer,
            "links": links
        },
        "source": formatted_sources
    }



# from fastapi import FastAPI
# from pydantic import BaseModel
# from openai import OpenAI
# import pickle
# import numpy as np
# from sentence_transformers import SentenceTransformer
# from dotenv import load_dotenv
# import os
# import json

# # .env fayldan API kalitlarini yuklash
# load_dotenv()

# # OpenAI mijozini ishga tushurish
# client = OpenAI(api_key=os.getenv("API_KEY"))

# # Config fayldan assistant_id ni olish
# with open("config.json", "r") as f:
#     config = json.load(f)
# ASSISTANT_ID = config.get("assistant_id")

# # Embedding ma’lumotlarini yuklash
# with open("talim_embeddings.pkl", "rb") as f:
#     lines, embeddings_dict = pickle.load(f)

# lines, embeddings = embeddings_dict  # unpack tuple
# embeddings = np.array(embeddings)

# embeddings = np.array(embeddings_dict["embeddings"])

# # Savol modelini aniqlash
# class Query(BaseModel):
#     question: str

# # FastAPI ilovasini yaratish
# app = FastAPI()

# # Sentence transformer modelini yuklash
# embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# @app.post("/ask")
# def ask_question(query: Query):
#     if not ASSISTANT_ID:
#         return {"error": "❗ assistant_id config.json faylda topilmadi."}

#     # 1. Foydalanuvchi savolini embedding qilish
#     query_embedding = embedding_model.encode(query.question)

#     # 2. Cosine similarity bo‘yicha eng yaqin 3 ta matnni topish
#     scores = np.dot(embeddings, query_embedding) / (
#         np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding)
#     )
#     top_indices = np.argsort(scores)[::-1][:3]
#     top_matches = [(lines[i], float(scores[i])) for i in top_indices]

#     # 3. Kontekstga asoslangan prompt yaratish
#     context_prompt = (
#         "Quyidagi matn(lar) asosida savolga qisqa va aniq javob ber:\n\n"
#         + "\n\n---\n\n".join([text for text, _ in top_matches]) +
#         f"\n\nSavol: {query.question}\n\nJavob:"
#     )

#     # 4. Thread yaratish va foydalanuvchi savolini yuborish
#     thread = client.beta.threads.create()
#     client.beta.threads.messages.create(
#         thread_id=thread.id,
#         role="user",
#         content=context_prompt
#     )

#     # 5. Assistant ishga tushirish
#     run = client.beta.threads.runs.create_and_poll(
#         thread_id=thread.id,
#         assistant_id=ASSISTANT_ID
#     )

#     # 6. Javobni olish
#     messages = client.beta.threads.messages.list(thread_id=thread.id)
#     answer = messages.data[0].content[0].text.value.strip()

#     # 7. Eng yaqin mos kelgan manba va score
#     top_source, top_score = top_matches[0]

#     return {
#         "question": query.question,
#         "answer": answer,
#         "source": top_source,
#         "similarity": round(top_score, 3)
#     }
