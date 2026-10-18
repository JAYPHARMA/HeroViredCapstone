from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from resume_matcher.data import load_huggingface_dataset, load_local_csv
from resume_matcher.model import ResumeJDMatcher


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the resume-JD matching model.")
    parser.add_argument("--data", default="data/sample_resume_jd.csv", help="CSV path for local training data.")
    parser.add_argument("--source", choices=["local", "hf"], default="local", help="Use local CSV or Hugging Face dataset.")
    parser.add_argument("--output", default="models/sample_baseline.joblib", help="Separate baseline output; never overwrites the deployed capstone model.")
    parser.add_argument("--hf-split", default="train", help="Hugging Face split name.")
    parser.add_argument("--hf-test-split", default="test", help="Hugging Face evaluation split name.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.source == "hf":
        train_frame = load_huggingface_dataset(split=args.hf_split)
        holdout_frame = load_huggingface_dataset(split=args.hf_test_split)
    else:
        frame = load_local_csv(args.data)
        train_frame, holdout_frame = train_test_split(
            frame,
            test_size=0.25,
            random_state=42,
            stratify=frame["label"] if frame["label"].nunique() > 1 else None,
        )
    model = ResumeJDMatcher().fit(train_frame)
    model.save(args.output)

    Path("data/sample_processed").mkdir(parents=True, exist_ok=True)
    train_frame.to_csv("data/sample_processed/train.csv", index=False)
    holdout_frame.to_csv("data/sample_processed/holdout.csv", index=False)
    print(f"Trained on {len(train_frame)} rows; holdout rows: {len(holdout_frame)}")
    print(f"Saved model to {args.output}")


if __name__ == "__main__":
    main()
