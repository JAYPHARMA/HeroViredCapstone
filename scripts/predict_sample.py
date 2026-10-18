from __future__ import annotations

import json

import pandas as pd

from resume_matcher.model import ResumeJDMatcher


def main() -> None:
    model = ResumeJDMatcher.load("models/resume_matcher.joblib")
    frame = pd.read_csv("data/sample_resume_jd.csv")
    jd_text = frame.iloc[0]["jd_text"]
    candidates = frame[["candidate_name", "resume_text"]].head(3).to_dict(orient="records")
    ranked = model.rank_candidates(jd_text, candidates)
    print(json.dumps(ranked, indent=2))


if __name__ == "__main__":
    main()
