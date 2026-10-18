"""Download one pinned encoder snapshot and save a self-contained local copy."""

import json
from pathlib import Path
from huggingface_hub import model_info
from sentence_transformers import SentenceTransformer
from resume_matcher.advanced import ENCODER_DIR, ENCODER_ID


def main():
    manifest = ENCODER_DIR.parent / "encoder_manifest.json"
    if manifest.exists() and (ENCODER_DIR / "modules.json").exists():
        print("Local encoder already ready:", manifest.read_text())
        return
    revision = model_info(ENCODER_ID).sha
    model = SentenceTransformer(ENCODER_ID, revision=revision, device="cpu")
    model.max_seq_length = 256
    model.save(str(ENCODER_DIR))
    manifest.write_text(
        json.dumps(
            {
                "model": ENCODER_ID,
                "revision": revision,
                "dimensions": 384,
                "max_sequence_length": 256,
                "chunk_words": 150,
                "max_chunks": 8,
                "fine_tuned_on_capstone": False,
            },
            indent=2,
        )
    )
    print("Saved local encoder", manifest.read_text())


if __name__ == "__main__":
    main()
