import json
import os

CONFIG_PATH = "config.json"

def load_config():
    if not os.path.exists(CONFIG_PATH):
        return {"assistant_id": "", "vector_store_id": "", "file_ids": []}
    with open(CONFIG_PATH, "r") as f:
        data = json.load(f)
    return (
        data.get("assistant_id", ""),
        data.get("vector_store_id", ""),
        data.get("file_ids", [])
    )

def save_config(data: dict):
    with open(CONFIG_PATH, "w") as f:
        json.dump(data, f, indent=2)
