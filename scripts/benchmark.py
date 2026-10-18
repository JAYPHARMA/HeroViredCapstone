"""Measure local inference latency with one request at a time."""

import platform, time
import numpy as np
import pandas as pd
import torch
from resume_matcher.model import ResumeJDMatcher
from resume_matcher.research import ROOT, write_json


def main():
    model = ResumeJDMatcher.load(ROOT / "models/resume_matcher.joblib")
    test = pd.read_csv(ROOT / "data/processed/test.csv")
    indices = np.linspace(0, len(test) - 1, 20, dtype=int)
    first = test.iloc[indices[0]]
    model.predict_one(first.resume_text, first.jd_text)
    times = []
    for index in indices:
        row = test.iloc[index]
        model.embedding_cache = {}
        start = time.perf_counter()
        model.predict_one(row.resume_text, row.jd_text)
        times.append((time.perf_counter() - start) * 1000)
    result = {
        "samples": 20,
        "median_ms": float(np.median(times)),
        "p95_ms": float(np.percentile(times, 95)),
        "scope": "Standalone process, loaded encoder, empty document cache per pair, includes embedding generation and skill suggestions, excludes startup and HTTP/network; one inference request at a time.",
        "sample_indices": indices.tolist(),
        "cpu": platform.processor(),
        "python": platform.python_version(),
        "torch_threads": torch.get_num_threads(),
    }
    write_json(ROOT / "reports/latency.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    main()
