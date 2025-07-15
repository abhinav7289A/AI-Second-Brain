import argparse, json

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--raw", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    # Simple formatter: split into lines and prefix bullets
    lines = args.raw.split("\n")
    formatted = "\n".join(f"- {line.strip()}" for line in lines if line.strip())

    with open(args.output, "w") as f:
        json.dump({"formatted": formatted}, f)
