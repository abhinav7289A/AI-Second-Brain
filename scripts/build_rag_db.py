import os
import json
from glob import glob
from tqdm import tqdm

import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

# --- MODIFIED SECTION START ---

# 1) Configure a persistent Chroma client & collection
# This will save the database to a directory named 'chroma_db'
# inside your 'data' folder.
db_path = os.path.join("data", "chroma_db")
os.makedirs(db_path, exist_ok=True)
client = chromadb.PersistentClient(path=db_path)

# Use get_or_create_collection to either create a new collection
# or load an existing one. This prevents errors on subsequent runs.
collection = client.get_or_create_collection(
    name="ai_second_brain",
    metadata={"description": "Multimodal docs for AI Second Brain"}
)

# --- MODIFIED SECTION END ---


# 2) Load text embedding model (Sentence-Transformer)
text_model = SentenceTransformer("all-MiniLM-L6-v2")  # fast, small

# 3) Load CLIP for images
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def chunk_text(text, max_len=500, overlap=50):
    """
    Splits text into chunks of ~max_len chars with given overlap.
    """
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + max_len, length)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += max_len - overlap
    return chunks

# … inside build_rag_db.py …

def embed_and_add(item_id, embedding, metadata):
    """
    Adds an item to ChromaDB collection, cleaning None → "".
    """
    # clean metadata: only keep primitives, replace None
    clean_meta = {}
    for k, v in metadata.items():
        if v is None:
            clean_meta[k] = ""
        elif isinstance(v, (bool, int, float, str)):
            clean_meta[k] = v
        else:
            # if it's something else, convert to string
            clean_meta[k] = str(v)

    collection.add(
        ids=[item_id],
        embeddings=[embedding],
        metadatas=[clean_meta],
        documents=[clean_meta.get("text_preview", "")]
    )


def process_corpus(corpus_path):
    # Check if the corpus file exists to avoid errors
    if not os.path.exists(corpus_path):
        print(f"[build_rag] Error: Corpus file not found at {corpus_path}")
        # Create an empty file to prevent crashing the rest of the script logic
        with open(corpus_path, "w") as f:
            pass # an empty file
        return

    with open(corpus_path, "r", encoding="utf-8") as f:
        # Read all lines to prevent TQDM from erroring on an empty file
        lines = f.readlines()
        if not lines:
            print("[build_rag] Corpus file is empty. Nothing to process.")
            return

        for idx, line in enumerate(tqdm(lines, desc="Reading corpus")):
            try:
                doc = json.loads(line)
            except json.JSONDecodeError:
                print(f"[build_rag] Warning: Skipping malformed JSON on line {idx+1}")
                continue

            src_type = doc.get("source_type")
            src_file = doc.get("source_file")
            page_seg = doc.get("page_or_segment")
            text = doc.get("text")
            meta = {
                "source_type": src_type,
                "source_file": src_file,
                "page_or_segment": page_seg
            }

            # TEXT / AUDIO: chunk & embed
            if src_type in ("text", "audio"):
                chunks = chunk_text(text)
                for i, chunk in enumerate(chunks):
                    # Ensure embedding is a list of floats
                    emb = text_model.encode(chunk).tolist()
                    item_id = f"{idx}-{i}"
                    metadata = {**meta, "text_preview": chunk[:100]}
                    embed_and_add(item_id, emb, metadata)

            # IMAGE: embed via CLIP
            elif src_type == "image":
                img_path = doc.get("extra", {}).get("path")
                if not img_path:
                    print(f"[build_rag] Skipping image entry at index {idx} due to missing path.")
                    continue
                try:
                    image = Image.open(img_path).convert("RGB")
                    inputs = clip_processor(images=image, return_tensors="pt")
                    outputs = clip_model.get_image_features(**inputs)
                    # Ensure embedding is a list of floats
                    emb = outputs.detach().cpu().numpy()[0].tolist()
                    item_id = f"{idx}-img"
                    metadata = {**meta, "image_path": img_path}
                    embed_and_add(item_id, emb, metadata)
                except Exception as e:
                    print(f"[build_rag] Skipping image {img_path}: {e}")

            else:
                # skip unknown types
                continue

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    corpus_file = os.path.join("data", "corpus.jsonl")
    process_corpus(corpus_file)
    print(f"[build_rag] Done. Collection size: {collection.count()} vectors.")