import argparse
import json
import os
import time
from sentence_transformers import SentenceTransformer
import chromadb

# Paths
DB_PATH = os.path.join("data", "chroma_db")
COLLECTION = "ai_second_brain"
CACHE_FILE = os.path.join("data", "retrieve_cache.json")

# Load or init cache
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        cache = json.load(f)
else:
    cache = {}

# Initialize once
client = chromadb.PersistentClient(path=DB_PATH)
coll   = client.get_or_create_collection(name=COLLECTION)
model  = SentenceTransformer("all-MiniLM-L6-v2")

def save_cache():
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

def retrieve(question: str):
    # Return cached?
    if question in cache:
        return cache[question], 0.0

    start = time.time()
    q_emb = model.encode(question).tolist()
    res = coll.query(query_embeddings=[q_emb], n_results=5)
    context = "\n".join(f"[{i+1}] {d}" for i, d in enumerate(res["documents"][0]))
    elapsed = time.time() - start

    # Cache and persist
    cache[question] = context
    save_cache()
    return context, elapsed

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--question", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    try:
        context, dt = retrieve(args.question)
        out = {"context": context, "retrieval_time_s": round(dt, 3)}
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[retrieve_context] ERROR: {e}", flush=True)
        raise
