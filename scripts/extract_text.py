import fitz  # PyMuPDF
import os
import json

def extract_pdf_text(pdf_path):
    """
    Extracts text from each page of the PDF and returns a list of dicts:
    [{"page": int, "text": str}, ...]
    """
    doc = fitz.open(pdf_path)
    text_chunks = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text("text")
        if text.strip():
            text_chunks.append({"page": page_num + 1, "text": text})
    return text_chunks

def main():
    base_dir = os.path.dirname(__file__)
    input_dir = os.path.join(base_dir, "..", "data", "raw_pdfs")
    output_dir = os.path.join(base_dir, "..", "data", "processed_text")
    os.makedirs(output_dir, exist_ok=True)

    for fname in os.listdir(input_dir):
        if not fname.lower().endswith(".pdf"):
            continue
        pdf_path = os.path.join(input_dir, fname)
        chunks = extract_pdf_text(pdf_path)
        out_fname = os.path.splitext(fname)[0] + ".json"
        out_path = os.path.join(output_dir, out_fname)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
        print(f"[extract_text] Processed {fname} -> {out_path}")

if __name__ == "__main__":
    main()
