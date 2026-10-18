"""Full reproducible experiment: audit, grouped validation, selection, refit, test.

Never chooses a model using test results. Default local inputs are archived public
data, not the 12 hand-written demo examples. No capstone transformer fine-tuning.
"""

from __future__ import annotations
import argparse, copy, json, platform, time
from datetime import datetime, timezone
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    ExtraTreesClassifier,
    RandomForestClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from resume_matcher.advanced import EmbeddingResumeJDMatcher, encode_documents
from resume_matcher.data import DATASET_ID, DATASET_REVISION, load_huggingface_dataset
from resume_matcher.model import ResumeJDMatcher, LABEL_ORDER
from resume_matcher.preprocessing import clean_text
from resume_matcher.research import (
    ROOT,
    audit_frame,
    clean_training_data,
    grouped_split,
    frame_keys,
    evaluate,
    priority_scores,
    sha256,
    write_json,
)


def log(message):
    print(message, flush=True)


def ordered_probabilities(model, x):
    probs = model.classifier.predict_proba(x)
    labels = list(model.label_encoder.classes_)
    return probs[:, [labels.index(label) for label in LABEL_ORDER]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--download",
        action="store_true",
        help="Reload pinned public dataset snapshot instead of archived CSVs",
    )
    args = parser.parse_args()
    rawdir = ROOT / "data/raw"
    out = ROOT / "data/processed"
    reports = ROOT / "reports"
    if args.download:
        for split, dest in [
            ("train", "original_train.csv"),
            ("test", "original_test.csv"),
        ]:
            load_huggingface_dataset(split=split).to_csv(rawdir / dest, index=False)
    source = pd.read_csv(rawdir / "original_train.csv")
    test = pd.read_csv(rawdir / "original_test.csv")
    clean, quarantine = clean_training_data(source)
    train, val = grouped_split(clean)
    for name, frame in [
        ("train", train),
        ("validation", val),
        ("test", test),
        ("train_full", clean),
        ("quarantined", quarantine),
    ]:
        frame.to_csv(out / f"{name}.csv", index=False)
    # Keep holdout.csv compatible with original evaluation commands.
    test.to_csv(out / "holdout.csv", index=False)
    audits = {
        name: audit_frame(frame)
        for name, frame in [
            ("original_train", source),
            ("clean_train_full", clean),
            ("train", train),
            ("validation", val),
            ("test", test),
        ]
    }
    trainkeys = set(frame_keys(clean))
    testkeys = frame_keys(test)
    audits["splits"] = {
        "test_pairs_also_in_training": int(testkeys.isin(trainkeys).sum()),
        "test_jds_also_in_training": int(
            test.jd_text.map(clean_text).isin(set(clean.jd_text.map(clean_text))).sum()
        ),
        "test_resumes_also_in_training": int(
            test.resume_text.map(clean_text)
            .isin(set(clean.resume_text.map(clean_text)))
            .sum()
        ),
        "train_validation_jd_overlap": 0,
        "quarantined_rows": len(quarantine),
        "policy": "Quarantine conflicting normalized pairs in source train only, then deduplicate. Original test remains untouched.",
    }
    write_json(reports / "data_audit.json", audits)
    log(
        f"Splits: training={len(train)}, validation={len(val)}, final training={len(clean)}, untouched test={len(test)}"
    )

    baseline = ResumeJDMatcher().fit(train)
    xb_train = baseline._feature_matrix(train.resume_text, train.jd_text)
    xb_val = baseline._feature_matrix(val.resume_text, val.jd_text)
    ytrain = baseline.label_encoder.transform(train.label)
    baseline_predictions = baseline.label_encoder.inverse_transform(
        baseline.classifier.predict(xb_val)
    )
    experiments = [
        {
            "model": "TF-IDF + Random Forest baseline",
            "parameters": baseline.classifier.get_params(),
            "validation_accuracy": accuracy_score(val.label, baseline_predictions),
            "validation_macro_f1": f1_score(
                val.label, baseline_predictions, average="macro"
            ),
            "training_seconds": None,
        }
    ]

    cache_path = ROOT / "models/document_embeddings.joblib"
    texts = list(
        dict.fromkeys(
            pd.concat(
                [clean.resume_text, clean.jd_text, test.resume_text, test.jd_text]
            ).map(clean_text)
        )
    )
    encoder_hash = sha256(ROOT / "models/encoder_manifest.json")
    cache = joblib.load(cache_path) if cache_path.exists() else {}
    if cache.get("encoder_manifest_sha256") != encoder_hash:
        cache = {}
    lookup = cache.get("embeddings", {})
    missing = [t for t in texts if t not in lookup]
    if missing:
        log(
            f"Encoding {len(missing)} unique documents with frozen Sentence-BERT (no labels)..."
        )
        lookup.update(zip(missing, encode_documents(missing, show_progress=True)))
        joblib.dump(
            {"encoder_manifest_sha256": encoder_hash, "embeddings": lookup},
            cache_path,
            compress=3,
        )
    advanced = EmbeddingResumeJDMatcher()
    advanced.vectorizer = baseline.vectorizer
    advanced.label_encoder = baseline.label_encoder
    advanced.embedding_cache = lookup
    xa_train = advanced._feature_matrix(train.resume_text, train.jd_text)
    xa_val = advanced._feature_matrix(val.resume_text, val.jd_text)
    configurations = [
        (
            "Embedding cosine + logistic",
            make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=1, class_weight="balanced", max_iter=1500, random_state=42
                ),
            ),
            [0],
        ),
        (
            "Sentence-BERT + logistic C0.1",
            make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=0.1, class_weight="balanced", max_iter=1500, random_state=42
                ),
            ),
            None,
        ),
        (
            "Sentence-BERT + logistic C1",
            make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=1, class_weight="balanced", max_iter=1500, random_state=42
                ),
            ),
            None,
        ),
        (
            "Sentence-BERT + ExtraTrees leaf2",
            ExtraTreesClassifier(
                n_estimators=350,
                min_samples_leaf=2,
                max_features=0.5,
                class_weight="balanced",
                random_state=42,
                n_jobs=4,
            ),
            None,
        ),
        (
            "Sentence-BERT + ExtraTrees leaf5",
            ExtraTreesClassifier(
                n_estimators=350,
                min_samples_leaf=5,
                max_features=0.5,
                class_weight="balanced",
                random_state=42,
                n_jobs=4,
            ),
            None,
        ),
        (
            "Sentence-BERT + RandomForest leaf3",
            RandomForestClassifier(
                n_estimators=300,
                min_samples_leaf=3,
                max_features=0.3,
                class_weight="balanced",
                random_state=42,
                n_jobs=4,
            ),
            None,
        ),
    ]
    fitted = []
    for name, estimator, subset in configurations:
        start = time.perf_counter()
        xtr = xa_train[:, subset] if subset else xa_train
        xv = xa_val[:, subset] if subset else xa_val
        estimator.fit(xtr, ytrain)
        pred = baseline.label_encoder.inverse_transform(estimator.predict(xv))
        row = {
            "model": name,
            "parameters": {k: str(v) for k, v in estimator.get_params().items()},
            "validation_accuracy": float(accuracy_score(val.label, pred)),
            "validation_macro_f1": float(f1_score(val.label, pred, average="macro")),
            "training_seconds": round(time.perf_counter() - start, 3),
        }
        experiments.append(row)
        fitted.append((row, estimator, subset))
        log(
            f"{name}: validation macro F1={row['validation_macro_f1']:.4f}, accuracy={row['validation_accuracy']:.4f}"
        )
    # Embedding-cosine is a comparison baseline; choose a full paired model.
    selected, estimator, _ = max(
        [r for r in fitted if r[2] is None], key=lambda r: r[0]["validation_macro_f1"]
    )
    log("Selected by validation macro F1: " + selected["model"])
    write_json(reports / "experiments/experiments.json", experiments)
    pd.DataFrame(
        [{k: v for k, v in r.items() if k != "parameters"} for r in experiments]
    ).to_csv(reports / "experiments/experiments.csv", index=False)
    curves = []
    rng = np.random.default_rng(42)
    # Stratified fixed-row subsets; validation JDs stay disjoint at all fractions.
    for fraction in [0.25, 0.5, 1.0]:
        indices = []
        for label in np.unique(ytrain):
            pool = np.flatnonzero(ytrain == label)
            indices.extend(
                rng.choice(pool, max(1, int(len(pool) * fraction)), replace=False)
            )
        candidate = clone(estimator).fit(xa_train[indices], ytrain[indices])
        curves.append(
            {
                "fraction": fraction,
                "rows": len(indices),
                "train_macro_f1": float(
                    f1_score(
                        ytrain[indices],
                        candidate.predict(xa_train[indices]),
                        average="macro",
                    )
                ),
                "validation_macro_f1": float(
                    f1_score(
                        baseline.label_encoder.transform(val.label),
                        candidate.predict(xa_val),
                        average="macro",
                    )
                ),
            }
        )
    write_json(reports / "experiments/learning_curve.json", curves)

    # Final refit uses source train only, after the configuration has been selected.
    final_baseline = ResumeJDMatcher().fit(clean)
    final_baseline.save(ROOT / "models/baseline.joblib")
    final = EmbeddingResumeJDMatcher(clone(estimator))
    final.embedding_cache = lookup
    final.fit(clean)
    final.metadata = {
        "selected_model": selected["model"],
        "selection_metric": "validation_macro_f1",
        "validation_macro_f1": selected["validation_macro_f1"],
        "encoder_frozen": True,
        "dataset_revision": DATASET_REVISION,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    final.save(ROOT / "models/resume_matcher.joblib")
    artifacts = {}
    for name, model in [
        ("majority", None),
        ("baseline", final_baseline),
        ("advanced", final),
    ]:
        if model is None:
            majority = clean.label.mode()[0]
            predictions = np.repeat(majority, len(test))
            scores = np.zeros(len(test))
            probs = np.tile([1.0, 0.0, 0.0], (len(test), 1))
        else:
            features = model._feature_matrix(test.resume_text, test.jd_text)
            probs = ordered_probabilities(model, features)
            predictions = np.array(LABEL_ORDER)[probs.argmax(axis=1)]
            scores = priority_scores(
                probs,
                features[:, 0],
                features[:, 2] if name == "advanced" else features[:, 1],
            )
        artifacts[name] = evaluate(test, predictions, scores)
        result = pd.DataFrame(
            {
                "candidate_name": test.candidate_name,
                "actual": test.label,
                "predicted": predictions,
                "score": scores,
            }
        )
        result.to_csv(reports / f"{name}_test_predictions.csv", index=False)
        log(
            f"TEST {name}: macro F1={artifacts[name]['macro_f1']:.4f}, accuracy={artifacts[name]['accuracy']:.4f}"
        )
    # Inference benchmark includes actual embedding computation, not training cache hits.
    final.embedding_cache = {}
    final.predict_one(test.iloc[0].resume_text, test.iloc[0].jd_text)
    latency = []
    for row in test.head(20).itertuples():
        final.embedding_cache = {}
        start = time.perf_counter()
        final.predict_one(row.resume_text, row.jd_text)
        latency.append(1000 * (time.perf_counter() - start))
    benchmark = {
        "samples": 20,
        "median_ms": float(np.median(latency)),
        "p95_ms": float(np.percentile(latency, 95)),
        "scope": "CPU, loaded encoder, empty document cache for each pair, includes skill suggestions, excludes HTTP/network and process/model startup",
        "cpu": platform.processor(),
        "python": platform.python_version(),
    }
    write_json(reports / "latency.json", benchmark)
    manifest = {
        "dataset": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "seed": 42,
        "raw_sha256": {p.name: sha256(p) for p in rawdir.glob("original_*.csv")},
        "model_sha256": sha256(ROOT / "models/resume_matcher.joblib"),
        "model": final.metadata,
        "splits": {
            n: len(f)
            for n, f in [
                ("train", train),
                ("validation", val),
                ("train_full", clean),
                ("test", test),
            ]
        },
        "test_evaluation_policy": "Configuration chosen on validation macro F1, then refit on clean source train. Test results do not select a model.",
    }
    write_json(reports / "run_manifest.json", manifest)
    write_json(reports / "metrics.json", artifacts)
    log("Completed. Generate reports/notebooks with scripts/build_submission.py")


if __name__ == "__main__":
    main()
