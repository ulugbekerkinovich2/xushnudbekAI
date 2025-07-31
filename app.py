from openai import OpenAI
from dotenv import load_dotenv
import os
import json


load_dotenv()
client = OpenAI(api_key=os.getenv("API_KEY"))

vector_store = client.vector_stores.create(
    name='Education'
)

client.vector_stores.files.upload_and_poll(
    vector_store_id=vector_store.id,
    file=open("talim.txt", "rb")
)
