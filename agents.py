import os
import json
import subprocess
import tempfile
import sys

# --- paths to your agent scripts (in agents/) ---
AGENTS_DIR = "agents"
DECOMP = os.path.join(AGENTS_DIR, "query_decomposer.py")
RETRV  = os.path.join(AGENTS_DIR, "retrieve_context.py")
GEN    = os.path.join(AGENTS_DIR, "generate_answer.py")
FMT    = os.path.join(AGENTS_DIR, "format_answer.py")

def call_agent(script, args):
    """
    Run the agent script using the same Python interpreter,
    so it picks up venv‑installed packages.
    """
    cmd = [sys.executable, script]
    for k, v in args.items():
        cmd += [f"--{k}", v]
    subprocess.run(cmd, check=True)
    return json.load(open(args["output"], encoding="utf-8"))

def main():
    query = input("Enter your question: ").strip()
    if not query:
        print("No question provided.")
        return

    with tempfile.TemporaryDirectory() as tmp:
        # 1) Decompose
        dec_file = os.path.join(tmp, "decomposed.json")
        dec = call_agent(DECOMP, {"input": query, "output": dec_file})
        primary_q = dec["primary"]

        # 2) Retrieve
        ctx_file = os.path.join(tmp, "context.json")
        ctx = call_agent(RETRV, {"question": primary_q, "output": ctx_file})
        context = ctx["context"]

        # 3) Generate
        raw_file = os.path.join(tmp, "raw_answer.json")
        raw = call_agent(GEN, {"question": primary_q, "context": context, "output": raw_file})
        raw_answer = raw["answer"]

        # 4) Format
        fmt_file = os.path.join(tmp, "formatted.json")
        fmt = call_agent(FMT, {"raw": raw_answer, "output": fmt_file})
        formatted = fmt["formatted"]

    print("\n--- Final Answer ---\n")
    print(formatted)

if __name__ == "__main__":
    main()
