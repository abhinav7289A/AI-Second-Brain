import os
import json
from glob import glob

def load_text_chunks(text_dir):
    for path in glob(os.path.join(text_dir, "*.json")):
        fname = os.path.basename(path)
        with open(path, 'r', encoding='utf-8') as f:
            pages = json.load(f)
        for page in pages:
            yield {
                "source_type": "text",
                "source_file": fname,
                "page_or_segment": page["page"],
                "text": page["text"].replace("\n", " ").strip(),
                "extra": {}
            }

def load_audio_segments(audio_dir):
    for path in glob(os.path.join(audio_dir, "*.json")):
        fname = os.path.basename(path)
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # full transcript as one chunk
        yield {
            "source_type": "audio",
            "source_file": fname,
            "page_or_segment": None,
            "text": data["transcript"].replace("\n", " ").strip(),
            "extra": {
                "duration": data.get("duration"),
            }
        }
        # individual segments
        for idx, seg in enumerate(data.get("segments", []), start=1):
            yield {
                "source_type": "audio",
                "source_file": fname,
                "page_or_segment": idx,
                "text": seg["text"].replace("\n", " ").strip(),
                "extra": {
                    "start": seg["start"],
                    "end": seg["end"]
                }
            }

def load_images(image_dir):
    for img_path in glob(os.path.join(image_dir, "*.*")):
        fname = os.path.basename(img_path)
        yield {
            "source_type": "image",
            "source_file": fname,
            "page_or_segment": None,
            "text": f"[Image: {fname}]",  # placeholder text to embed
            "extra": {
                "path": img_path
            }
        }

def main():
    base = os.path.dirname(__file__)
    text_dir  = os.path.join(base, "..", "data", "processed_text")
    audio_dir = os.path.join(base, "..", "data", "processed_audio")
    image_dir = os.path.join(base, "..", "data", "diagrams")
    out_path  = os.path.join(base, "..", "data", "corpus.jsonl")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as outf:
        for doc in load_text_chunks(text_dir):
            outf.write(json.dumps(doc, ensure_ascii=False) + "\n")
        for doc in load_audio_segments(audio_dir):
            outf.write(json.dumps(doc, ensure_ascii=False) + "\n")
        for doc in load_images(image_dir):
            outf.write(json.dumps(doc, ensure_ascii=False) + "\n")

    print(f"[normalize] Wrote unified corpus to {out_path}")

if __name__ == "__main__":
    main()
