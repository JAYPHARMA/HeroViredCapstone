"""Skill extraction and skill-gap analysis.

This module intentionally uses a transparent keyword taxonomy. For a capstone
prototype, recruiters can inspect why a skill was detected and learners can
extend the taxonomy without retraining the model.
"""

from __future__ import annotations

import re
from functools import lru_cache
from dataclasses import dataclass

from .preprocessing import clean_text


SKILL_TAXONOMY: dict[str, tuple[str, ...]] = {
    "python": ("python",),
    "sql": ("sql", "postgresql", "mysql", "database"),
    "excel": ("excel", "spreadsheet"),
    "power bi": ("power bi", "powerbi"),
    "tableau": ("tableau",),
    "machine learning": ("machine learning", "supervised learning", "classification", "regression", "predictive modeling"),
    "deep learning": ("deep learning", "tensorflow", "keras", "pytorch", "neural"),
    "nlp": ("nlp", "natural language", "text classification", "transformers", "bert"),
    "scikit-learn": ("scikit-learn", "sklearn"),
    "pandas": ("pandas",),
    "statistics": ("statistics", "statistical", "a/b testing", "experimentation"),
    "feature engineering": ("feature engineering", "features"),
    "model evaluation": ("model evaluation", "f1", "accuracy", "precision", "recall", "confusion matrix"),
    "model explainability": ("explainability", "shap", "lime", "interpretability"),
    "rest api": ("rest api", "rest", "fastapi", "flask"),
    "docker": ("docker", "container"),
    "kubernetes": ("kubernetes", "k8s"),
    "ci/cd": ("ci/cd", "continuous integration", "pipeline"),
    "cloud deployment": ("cloud", "aws", "azure", "gcp", "deployment"),
    "java": ("java", "spring boot"),
    "javascript": ("javascript", "react", "node"),
    "testing": ("unit testing", "testing", "pytest", "junit"),
    "monitoring": ("monitoring", "observability", "logging"),
    "analytics": ("analytics", "dashboard", "dashboards", "kpi", "reporting"),
    "hr analytics": ("hr analytics", "workforce analytics", "attrition", "compensation"),
    "communication": ("communication", "stakeholder", "presentation", "collaborate"),
    "ats": ("ats", "applicant tracking"),
}


@dataclass(frozen=True)
class SkillGap:
    jd_skills: list[str]
    resume_skills: list[str]
    matched_skills: list[str]
    missing_skills: list[str]
    extra_skills: list[str]
    overlap_ratio: float


def _contains_phrase(text: str, phrase: str) -> bool:
    escaped = re.escape(phrase.lower())
    return re.search(rf"(?<![a-z0-9]){escaped}(?![a-z0-9])", text) is not None


def extract_skills(text: object) -> list[str]:
    cleaned = clean_text(text)
    return list(_extract_cleaned(cleaned))


@lru_cache(maxsize=4096)
def _extract_cleaned(cleaned: str) -> tuple[str, ...]:
    found: list[str] = []
    for canonical, aliases in SKILL_TAXONOMY.items():
        if any(_contains_phrase(cleaned, alias) for alias in aliases):
            found.append(canonical)
    return tuple(sorted(found))


def analyze_skill_gap(resume_text: object, jd_text: object) -> SkillGap:
    resume_skills = set(extract_skills(resume_text))
    jd_skills = set(extract_skills(jd_text))
    matched = sorted(resume_skills & jd_skills)
    missing = sorted(jd_skills - resume_skills)
    extra = sorted(resume_skills - jd_skills)
    overlap = len(matched) / max(len(jd_skills), 1)
    return SkillGap(
        jd_skills=sorted(jd_skills),
        resume_skills=sorted(resume_skills),
        matched_skills=matched,
        missing_skills=missing,
        extra_skills=extra,
        overlap_ratio=round(overlap, 4),
    )
