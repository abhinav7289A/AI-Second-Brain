import os
import json
import subprocess

from sentence_transformers import SentenceTransformer
import chromadb

# 1) Connect to persistent RAG DB
db_path = os.path.join("data", "chroma_db")
client = chromadb.PersistentClient(path=db_path)
collection = client.get_or_create_collection(name="ai_second_brain")

# 2) Load embedding model
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# 3) Config
LLM_MODEL = "llama3"   # change to your Ollama model name
TOP_K = 5                           # how many chunks to retrieve
CONTEXT_TOKEN_LIMIT = 1500          # approximate max context length

def retrieve_context(question: str):
    # embed the question
    q_emb = embed_model.encode(question).tolist()
    res = collection.query(query_embeddings=[q_emb], n_results=TOP_K)
    metas = res["metadatas"][0]
    docs  = res["documents"][0]
    # build a single context string, numbered
    context = []
    for i, (m, d) in enumerate(zip(metas, docs), 1):
        src = f"{m['source_type']}:{m['source_file']} pg/seg {m['page_or_segment']}"
        preview = d.replace("\n", " ")
        context.append(f"[{i}] {src} — {preview}")
    # join and truncate if too long
    full = "\n".join(context)
    if len(full) > CONTEXT_TOKEN_LIMIT:
        full = full[:CONTEXT_TOKEN_LIMIT] + "\n[...context truncated]"
    return full

def generate_answer(question: str, context: str):
    # craft prompt
    prompt = (
        "You are an academic research assistant. Use the following context excerpts to answer the question.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer in a clear, detailed manner, citing the context indices when helpful."
    )
    # call Ollama: prompt is positional
    result = subprocess.run(
        ["ollama", "run", LLM_MODEL, prompt],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print("❌ Ollama error:", result.stderr)
        return ""
    return result.stdout.strip()

def main():
    while True:
        q = input("\nEnter your question (or 'exit'): ").strip()
        if q.lower() in ("exit", "quit"):
            break
        ctx = retrieve_context(q)
        print("\n--- Retrieved Context ---")
        print(ctx)
        print("\n--- Generating Answer ---")
        ans = generate_answer(q, ctx)
        print(ans)

if __name__ == "__main__":
    main()
