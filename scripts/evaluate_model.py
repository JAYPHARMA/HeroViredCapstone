"""Evaluate a saved matcher without overwriting the narrative report."""

import argparse
import json
import joblib
import numpy as np
from pathlib import Path
from resume_matcher.data import load_local_csv
from resume_matcher.model import ResumeJDMatcher, LABEL_ORDER
from resume_matcher.advanced import EmbeddingResumeJDMatcher
from resume_matcher.research import ROOT, evaluate, priority_scores, write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=str(ROOT / "models/resume_matcher.joblib"))
    p.add_argument("--data", default=str(ROOT / "data/processed/test.csv"))
    p.add_argument("--report", default=str(ROOT / "reports/evaluation_recheck.json"))
    args = p.parse_args()
    model = ResumeJDMatcher.load(args.model)
    frame = load_local_csv(args.data)
    cache = ROOT / "models/document_embeddings.joblib"
    if isinstance(model, EmbeddingResumeJDMatcher) and cache.exists():
        model.embedding_cache = joblib.load(cache)["embeddings"]
    x = model._feature_matrix(frame.resume_text, frame.jd_text)
    probabilities = model.classifier.predict_proba(x)
    probabilities = probabilities[
        :, [list(model.label_encoder.classes_).index(label) for label in LABEL_ORDER]
    ]
    predictions = np.array(LABEL_ORDER)[probabilities.argmax(axis=1)]
    result = evaluate(
        frame,
        predictions,
        priority_scores(
            probabilities,
            x[:, 0],
            x[:, 2] if isinstance(model, EmbeddingResumeJDMatcher) else x[:, 1],
        ),
    )
    write_json(args.report, result)
    print(json.dumps(result, indent=2))
    print("Wrote", args.report)


if __name__ == "__main__":
    main()
