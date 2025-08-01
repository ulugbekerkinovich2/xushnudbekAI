# from openai import OpenAI
# from dotenv import load_dotenv
# import os
# import json
# from tqdm import tqdm
# import time
# import logging

# # Log sozlamalari
# logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')

# # API kalitni yuklash
# load_dotenv()
# client = OpenAI(api_key=os.getenv("API_KEY"))

# # Progres bar funksiyasi
# def loading_bar(msg, seconds=3):
#     logging.info(msg)
#     for _ in tqdm(range(30), desc="  ⏳", ncols=100):
#         time.sleep(seconds / 30)

# try:
#     # 1. Vector store yaratish
#     loading_bar("📦 Vector Store yaratilmoqda...")
#     vector_store = client.vector_stores.create(name="Initial-RAG-Vector")
#     vector_store_id = vector_store.id
#     logging.info(f"✅ Vector Store ID: {vector_store_id}")

#     # 2. Faylni yuklash
#     loading_bar("📁 Hujjat yuklanmoqda...")
#     file_path = "talim1.txt"
#     with open(file_path, "rb") as f:
#         uploaded_file = client.files.create(file=f, purpose="assistants")
#         file_id = uploaded_file.id
#     logging.info(f"✅ Fayl yuklandi. ID: {file_id}")

#     # 3. File ni vector store ga ulash
#     loading_bar("📌 Fayl vector store ga ulanmoqda...")
#     with open(file_path, "rb") as f:
#         file_batch = client.vector_stores.file_batches.upload_and_poll(
#             vector_store_id=vector_store_id,
#             files=[f]
#         )
#     logging.info(f"✅ File batch: {file_batch.id}")

#     # 4. Assistant yaratish
#     loading_bar("🤖 Assistent yaratilmoqda...")
#     assistant = client.beta.assistants.create(
#         name="Support Assistant",
#         instructions = 
#     "Siz ta'limga oid hujjatlar bo‘yicha savollarga javob beruvchi foydali yordamchisiz. "
#     "Javoblaringizni aniq va tushunarli tarzda bering. Har bir fakt yoki ma'lumot manbasi bilan birga ko‘rsatilishi kerak, masalan:   ko‘rinishida.",

#         model="gpt-3.5-turbo",  # yoki "gpt-3.5-turbo"
#         tools=[{"type": "file_search"}],
#         tool_resources={
#             "file_search": {
#                 "vector_store_ids": [vector_store_id]
#             }
#         }
#     )
#     assistant_id = assistant.id
#     logging.info(f"✅ Assistant ID: {assistant_id}")

#     # 5. config.json faylga saqlash
#     config = {
#         "vector_store_id": vector_store_id,
#         "assistant_id": assistant_id,
#         "file_ids": [file_id]
#     }
#     with open("config.json", "w") as f:
#         json.dump(config, f, indent=2)
#     logging.info("✅ Ma'lumotlar 'config.json' faylga saqlandi.")

# except Exception as e:
#     logging.error(f"❌ Xatolik yuz berdi: {e}")

from openai import OpenAI
from dotenv import load_dotenv
import os
import json
from tqdm import tqdm
import time
import logging

# Log sozlamalari
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')

# API kalitini yuklash
load_dotenv()
client = OpenAI(api_key=os.getenv("API_KEY"))

# Progres bar funksiyasi
def loading_bar(msg, seconds=3):
    logging.info(msg)
    for _ in tqdm(range(30), desc="  ⏳", ncols=100):
        time.sleep(seconds / 30)

try:
    # config.json dan mavjud assistant_id va vector_store_id ni yuklab olish
    with open("config.json", "r", encoding="utf-8") as f:
        config = json.load(f)

    assistant_id = config.get("assistant_id")
    vector_store_id = config.get("vector_store_id")
    file_ids = config.get("file_ids", [])

    if not assistant_id or not vector_store_id:
        raise ValueError("❌ assistant_id yoki vector_store_id topilmadi. Avval ularni yaratganingizga ishonch hosil qiling.")

    # 1. Faylni yuklash
    file_path = "talim1.txt"
    loading_bar("📁 Yangi hujjat yuklanmoqda...")
    with open(file_path, "rb") as f:
        uploaded_file = client.files.create(file=f, purpose="assistants")
        new_file_id = uploaded_file.id
        logging.info(f"✅ Yangi fayl yuklandi. ID: {new_file_id}")

    # 2. Vector store ga faylni ulang
    loading_bar("📌 Fayl vector store ga ulanmoqda...")
    with open(file_path, "rb") as f:
        file_batch = client.vector_stores.file_batches.upload_and_poll(
            vector_store_id=vector_store_id,
            files=[f]
        )
    logging.info(f"✅ File batch ID: {file_batch.id}")

    # 3. Assistantni yangilash
    loading_bar("🔄 Assistent yangilanmoqda...")
    updated_assistant = client.beta.assistants.update(
        assistant_id=assistant_id,
        instructions=(
            "Siz O‘zbekiston Respublikasi ta’lim sohasi hujjatlari asosida ishlovchi yordamchisiz. "
            "Faqat o‘zbek tilida aniq, lo‘nda va tushunarli javoblar bering. "
            "Javobda keltirilgan har bir fakt uchun albatta manba ko‘rsating. "
            "Agar mavjud bo‘lsa, hujjat havolasini (linkini) ham qo‘shing. "
            "Javob formatida manbani 'source:' degan so‘z bilan yakunlang va shu yerda ko‘rsating."
        ),
        tools=[{"type": "file_search"}],
        tool_resources={
            "file_search": {
                "vector_store_ids": [vector_store_id]
            }
        }
    )
    logging.info("✅ Assistant muvaffaqiyatli yangilandi.")

    # 4. config.json faylni yangilash
    updated_file_ids = list(set(file_ids + [new_file_id]))
    config["file_ids"] = updated_file_ids
    with open("config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    logging.info("✅ Yangi holat 'config.json' faylga saqlandi.")

except Exception as e:
    logging.error(f"❌ Xatolik yuz berdi: {e}")
