import os
import sys
import json
import subprocess
import tempfile
from glob import glob

import gradio as gr

# Paths (same as before)
AGENTS_DIR = "agents"
DATA_DIR   = "data"
RAW_PDF_DIR    = os.path.join(DATA_DIR, "raw_pdfs")
RAW_AUDIO_DIR  = os.path.join(DATA_DIR, "audio")
DIAGRAMS_DIR   = os.path.join(DATA_DIR, "diagrams")
PROCESSED_TEXT = os.path.join(DATA_DIR, "processed_text")
PROCESSED_AUDIO= os.path.join(DATA_DIR, "processed_audio")
CORPUS_FILE    = os.path.join(DATA_DIR, "corpus.jsonl")
RAG_SCRIPT     = os.path.join("scripts", "build_rag_db.py")

for d in (RAW_PDF_DIR, RAW_AUDIO_DIR, DIAGRAMS_DIR, PROCESSED_TEXT, PROCESSED_AUDIO):
    os.makedirs(d, exist_ok=True)

def ingest_and_reindex(files):
    did_pdf = did_audio = False
    for path in files:
        fname, ext = os.path.basename(path.name), os.path.splitext(path.name)[1].lower()
        if ext == ".pdf":
            dest, did_pdf = os.path.join(RAW_PDF_DIR, fname), True
        elif ext in (".mp3", ".wav", ".m4a"):
            dest, did_audio = os.path.join(RAW_AUDIO_DIR, fname), True
        elif ext in (".png", ".jpg", ".jpeg"):
            dest = os.path.join(DIAGRAMS_DIR, fname)
        else:
            continue
        with open(dest, "wb") as out, open(path.name, "rb") as inp:
            out.write(inp.read())

    if did_pdf:
        subprocess.run([sys.executable, "scripts/extract_text.py"], check=True)
        subprocess.run([sys.executable, "scripts/extract_images.py"], check=True)
    if did_audio:
        subprocess.run([sys.executable, "scripts/extract_audio.py"], check=True)

    subprocess.run([sys.executable, "scripts/normalize_data.py"], check=True)
    subprocess.run([sys.executable, RAG_SCRIPT], check=True)
    return "✅ Ingestion and reindexing complete."

def call_agent(script, args):
    cmd = [sys.executable, script]
    for k, v in args.items():
        cmd += [f"--{k}", v]
    subprocess.run(cmd, check=True)
    return json.load(open(args["output"], encoding="utf-8"))

def pipeline(query: str) -> str:
    if not query:
        return "Please enter a question."

    try:
        with tempfile.TemporaryDirectory() as tmp:
            # 1) Decompose
            dec = call_agent(
                os.path.join(AGENTS_DIR, "query_decomposer.py"),
                {"input": query, "output": os.path.join(tmp, "decomp.json")}
            )
            primary = dec["primary"]

            # 2) Retrieve (with timing)
            ctx_out = os.path.join(tmp, "ctx.json")
            ctx = call_agent(
                os.path.join(AGENTS_DIR, "retrieve_context.py"),
                {"question": primary, "output": ctx_out}
            )
            context = ctx.get("context", "")
            rt = ctx.get("retrieval_time_s")

            # 3) Generate
            raw = call_agent(
                os.path.join(AGENTS_DIR, "generate_answer.py"),
                {"question": primary, "context": context, "output": os.path.join(tmp, "raw.json")}
            )

            # 4) Format
            fmt = call_agent(
                os.path.join(AGENTS_DIR, "format_answer.py"),
                {"raw": raw.get("answer", ""), "output": os.path.join(tmp, "fmt.json")}
            )
            answer_text = fmt.get("formatted", "")

        # Append timing footer if available
        if rt is not None:
            answer_text += f"\n\n_(retrieval took {rt} s)_"

        return answer_text

    except Exception as e:
        # Surface any errors in the UI
        return f"❌ Error during pipeline execution: {e}"

def safe_pipeline(query):
    try:
        return pipeline(query)
    except Exception as e:
        return f"❌ Error: {e}"

with gr.Blocks() as demo:
    gr.Markdown("## AI Second Brain\nUpload materials, reindex, then ask questions.")

    with gr.Tab("Manage Corpus"):
        upload     = gr.File(file_count="multiple", label="Upload PDFs, audio, or images")
        ingest_btn = gr.Button("Ingest & Reindex")
        ingest_out = gr.Textbox(label="Status")
        ingest_btn.click(ingest_and_reindex, inputs=[upload], outputs=[ingest_out])

    with gr.Tab("Ask Questions"):
        with gr.Row():
            with gr.Column(scale=1):
                question = gr.Textbox(lines=4, placeholder="Enter your question…", label="Your Question")
                ask_btn  = gr.Button("Ask AI Second Brain")
            with gr.Column(scale=1):
                # The key change is adding max_lines=10
                # This makes the Textbox scrollable once content exceeds 10 lines.
                answer   = gr.Textbox(lines=10, max_lines=10, label="Answer", interactive=False)

        ask_btn.click(safe_pipeline, inputs=[question], outputs=[answer])

if __name__ == "__main__":
    demo.launch()