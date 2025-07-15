import os
import json
import whisper
from tqdm import tqdm

def transcribe_file(model, audio_path):
    """
    Transcribes a single audio file.
    Returns a dict: {
      "filename": <str>,
      "duration": <float>,
      "transcript": <str>,
      "segments": [ {start, end, text}, … ]
    }
    """
    result = model.transcribe(audio_path)
    return {
        "filename": os.path.basename(audio_path),
        "duration": result.get("duration", None),
        "transcript": result["text"].strip(),
        "segments": [
            {
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"].strip()
            }
            for seg in result.get("segments", [])
        ],
    }

def main():
    base_dir = os.path.dirname(__file__)
    audio_dir = os.path.join(base_dir, "..", "data", "audio")
    output_dir = os.path.join(base_dir, "..", "data", "processed_audio")
    os.makedirs(output_dir, exist_ok=True)

    # Load Whisper model once (you can switch to "small", "medium", etc.)
    model = whisper.load_model("base")

    for fname in tqdm(os.listdir(audio_dir), desc="Transcribing audio"):
        if not fname.lower().endswith((".mp3", ".wav", ".m4a")):
            continue
        audio_path = os.path.join(audio_dir, fname)
        print(f"\n→ Transcribing {fname} …")
        data = transcribe_file(model, audio_path)

        out_fname = os.path.splitext(fname)[0] + ".json"
        out_path = os.path.join(output_dir, out_fname)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"[extract_audio] Saved transcript to {out_path}")

if __name__ == "__main__":
    main()
