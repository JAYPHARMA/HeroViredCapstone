"""Training and inference logic for resume-JD matching."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder

from .preprocessing import clean_text, token_set
from .skills import analyze_skill_gap


LABEL_ORDER = ["low_fit", "potential_fit", "strong_fit"]


class ResumeJDMatcher:
    """Interpretable resume/JD matcher using TF-IDF similarity plus skill features."""

    def __init__(self) -> None:
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2), min_df=1, max_features=5000
        )
        self.classifier = RandomForestClassifier(
            n_estimators=250,
            random_state=42,
            class_weight="balanced",
            min_samples_leaf=1,
        )
        self.label_encoder = LabelEncoder()
        self.is_fitted = False

    def fit(self, frame: pd.DataFrame) -> "ResumeJDMatcher":
        data = self._validate_frame(frame)
        all_text = pd.concat(
            [data["resume_text"], data["jd_text"]], ignore_index=True
        ).map(clean_text)
        self.vectorizer.fit(all_text)
        features = self._feature_matrix(data["resume_text"], data["jd_text"])
        labels = self.label_encoder.fit_transform(data["label"])
        self.classifier.fit(features, labels)
        self.is_fitted = True
        return self

    def predict_one(
        self, resume_text: str, jd_text: str, candidate_name: str = "Candidate"
    ) -> dict:
        self._require_fit()
        features = self._feature_matrix([resume_text], [jd_text])
        probabilities = self.classifier.predict_proba(features)[0]
        label_probabilities = {
            label: float(probabilities[index])
            for index, label in enumerate(self.label_encoder.classes_)
        }
        prediction = self.label_encoder.inverse_transform(
            [int(np.argmax(probabilities))]
        )[0]
        gap = analyze_skill_gap(resume_text, jd_text)
        semantic_similarity = float(features[0, 0])
        score = self._score(label_probabilities, semantic_similarity, gap.overlap_ratio)
        return {
            "candidate_name": candidate_name,
            "predicted_fit": prediction,
            "match_score": round(score, 2),
            "semantic_similarity": round(semantic_similarity, 4),
            "skill_overlap_ratio": gap.overlap_ratio,
            "probabilities": {
                key: round(value, 4) for key, value in label_probabilities.items()
            },
            "skill_gap": asdict(gap),
            "recommendation": self._recommendation(prediction, gap.missing_skills),
        }

    def rank_candidates(self, jd_text: str, candidates: Iterable[dict]) -> list[dict]:
        predictions = [
            self.predict_one(
                resume_text=str(candidate.get("resume_text", "")),
                jd_text=jd_text,
                candidate_name=str(
                    candidate.get("candidate_name", f"Candidate {index + 1}")
                ),
            )
            for index, candidate in enumerate(candidates)
        ]
        return sorted(predictions, key=lambda item: item["match_score"], reverse=True)

    def save(self, path: str | Path) -> None:
        self._require_fit()
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, destination)

    @staticmethod
    def load(path: str | Path) -> "ResumeJDMatcher":
        model = joblib.load(path)
        if not isinstance(model, ResumeJDMatcher):
            raise TypeError(f"Unsupported model artifact at {path}")
        model._require_fit()
        return model

    def _feature_matrix(self, resumes: Iterable[str], jds: Iterable[str]) -> np.ndarray:
        resume_list = list(resumes)
        jd_list = list(jds)
        resume_vectors = self.vectorizer.transform(
            [clean_text(text) for text in resume_list]
        )
        jd_vectors = self.vectorizer.transform([clean_text(text) for text in jd_list])
        rows: list[list[float]] = []
        for index, (resume_text, jd_text) in enumerate(zip(resume_list, jd_list)):
            cosine = cosine_similarity(resume_vectors[index], jd_vectors[index])[0, 0]
            gap = analyze_skill_gap(resume_text, jd_text)
            resume_tokens = token_set(resume_text)
            jd_tokens = token_set(jd_text)
            shared_tokens = resume_tokens & jd_tokens
            token_overlap = len(shared_tokens) / max(len(jd_tokens), 1)
            resume_length = max(len(clean_text(resume_text).split()), 1)
            jd_length = max(len(clean_text(jd_text).split()), 1)
            rows.append(
                [
                    float(cosine),
                    gap.overlap_ratio,
                    len(gap.matched_skills),
                    len(gap.missing_skills),
                    len(gap.resume_skills),
                    len(gap.jd_skills),
                    token_overlap,
                    min(resume_length / jd_length, 3.0),
                ]
            )
        return np.array(rows, dtype=float)

    @staticmethod
    def _validate_frame(frame: pd.DataFrame) -> pd.DataFrame:
        required = {"resume_text", "jd_text", "label"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")
        data = frame.dropna(subset=["resume_text", "jd_text", "label"]).copy()
        data["label"] = data["label"].astype(str).str.strip()
        return data[data["label"].isin(LABEL_ORDER)]

    @staticmethod
    def _score(
        probabilities: dict[str, float],
        semantic_similarity: float,
        skill_overlap: float,
    ) -> float:
        expected = (
            round(probabilities.get("potential_fit", 0), 4)
            + 2 * round(probabilities.get("strong_fit", 0), 4)
        ) / 2
        cosine = max(0.0, min(1.0, round(semantic_similarity, 4)))
        return 100 * (0.30 * expected + 0.45 * cosine + 0.25 * skill_overlap)

    @staticmethod
    def _recommendation(prediction: str, missing_skills: list[str]) -> str:
        if prediction == "strong_fit":
            return "Shortlist for recruiter review; validate experience depth and project evidence."
        if prediction == "potential_fit":
            skills = (
                ", ".join(missing_skills[:4])
                if missing_skills
                else "role-specific depth"
            )
            return f"Consider after screening; probe gaps around {skills}."
        skills = (
            ", ".join(missing_skills[:4])
            if missing_skills
            else "core role requirements"
        )
        return f"Do not reject automatically; manual review needed, but visible gaps include {skills}."

    def _require_fit(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted. Run scripts/train_model.py first.")
