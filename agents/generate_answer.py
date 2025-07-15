import argparse, json, subprocess

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--question", required=True)
    p.add_argument("--context", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    prompt = (
        "You are an academic assistant. Use the context below to answer.\n\n"
        f"Context:\n{args.context}\n\n"
        f"Question: {args.question}\n\n"
        "Answer in detail:"
    )

    result = subprocess.run(
        ["ollama", "run", "llama3", prompt],
        capture_output=True, text=True
    )
    answer = result.stdout.strip()
    with open(args.output, "w") as f:
        json.dump({"answer": answer}, f)
