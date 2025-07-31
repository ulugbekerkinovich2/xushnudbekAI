from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()  # .env faylni yuklaydi

secret_key = os.getenv("API_KEY")

print(secret_key)

client = OpenAI(api_key=secret_key)

# 1. Vector store yaratish
vector_store = client.vector_stores.create(name="Talim")

# 2. Hujjat yuklash (masalan: 'help_doc.txt')
file = open("talim.txt", "rb")
client.vector_stores.file_batches.upload_and_poll(
    vector_store_id=vector_store.id,
    files=[file]
)


# Assistant RAG ko‘rinishda ishlashi uchun retrieval qo‘shamiz
assistant = client.beta.assistants.create(
    name="My Custom RAG Assistant",
    instructions="Answer questions based only on the provided documents.",
    tools=[{"type": "retrieval"}],
    model="gpt-4-1106-preview",
    file_ids=[file.id]  # yoki vector_store.file_ids
)


# Assistant RAG ko‘rinishda ishlashi uchun retrieval qo‘shamiz
assistant = client.beta.assistants.create(
    name="My Custom RAG Assistant",
    instructions="Answer questions based only on the provided documents.",
    tools=[{"type": "retrieval"}],
    model="gpt-4-1106-preview",
    file_ids=[file.id]  # yoki vector_store.file_ids
)


# Chat sessiyasi ochiladi
thread = client.beta.threads.create()

# Savol yuboriladi
client.beta.threads.messages.create(
    thread_id=thread.id,
    role="user",
    content="Bizning kompaniya qaytarib berish siyosati qanday?"
)

# Javob olish
run = client.beta.threads.runs.create_and_poll(
    thread_id=thread.id,
    assistant_id=assistant.id,
)

# Natijani olish
messages = client.beta.threads.messages.list(thread_id=thread.id)
for msg in messages.data:
    print(msg.content[0].text.value)
