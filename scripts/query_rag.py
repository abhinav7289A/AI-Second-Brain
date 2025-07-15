import os
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# 1) Connect to persistent DB
db_path = os.path.join("data", "chroma_db")
client = chromadb.PersistentClient(path=db_path)
collection = client.get_or_create_collection(name="ai_second_brain")

# 2) Load query embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

def query_and_print(text, k=3):
    q_emb = model.encode(text).tolist()
    results = collection.query(query_embeddings=[q_emb], n_results=k)
    for idx, (meta, doc) in enumerate(zip(results["metadatas"][0], results["documents"][0]), 1):
        print(f"\n=== Result {idx} ===")
        print("Source:", meta["source_type"], meta["source_file"], "pg/seg", meta["page_or_segment"])
        print("Preview:", doc)

if __name__ == "__main__":
    while True:
        q = input("\nEnter your query (or 'exit'): ").strip()
        if q.lower() in ("exit", "quit"):
            break
        query_and_print(q, k=5)
