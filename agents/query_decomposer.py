import argparse, json

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    # For now, we don’t decompose—just pass through
    decomposed = {
        "primary": args.input,
        "subquestions": []
    }
    with open(args.output, "w") as f:
        json.dump(decomposed, f)
