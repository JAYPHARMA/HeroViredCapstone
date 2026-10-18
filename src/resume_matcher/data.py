"""Dataset loading and normalization utilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


STANDARD_COLUMNS = ["candidate_name", "job_title", "resume_text", "jd_text", "label"]
DATASET_ID = "facehuggerapoorv/resume-jd-match"
DATASET_REVISION = "bfab7acecac062628b39e0dd0c59e72d675edc9c"


def load_local_csv(path: str | Path) -> pd.DataFrame:
    return normalize_columns(pd.read_csv(path))


def load_huggingface_dataset(dataset_name: str = "facehuggerapoorv/resume-jd-match", split: str = "train") -> pd.DataFrame:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install the optional 'datasets' package to load from Hugging Face.") from exc

    dataset = load_dataset(dataset_name, split=split, revision=DATASET_REVISION if dataset_name == DATASET_ID else None)
    return normalize_columns(dataset.to_pandas())


def normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Map common dataset column names into the project schema."""
    frame = _expand_combined_hf_text(frame)
    lower_map = {column.lower().strip(): column for column in frame.columns}
    candidates = {
        "resume_text": ["resume_text", "resume", "cv", "profile", "candidate_resume", "Resume"],
        "jd_text": ["jd_text", "job_description", "jd", "description", "Job Description"],
        "label": ["label", "fit", "match", "match_label", "category", "score"],
        "candidate_name": ["candidate_name", "name"],
        "job_title": ["job_title", "role", "title", "position"],
    }

    result = pd.DataFrame()
    for standard_name, options in candidates.items():
        source = next((lower_map.get(option.lower()) for option in options if option.lower() in lower_map), None)
        if source is not None:
            result[standard_name] = frame[source]

    if "label" in result:
        result["label"] = result["label"].map(_normalize_label)
    if "candidate_name" not in result:
        result["candidate_name"] = [f"Candidate {index + 1}" for index in range(len(frame))]
    if "job_title" not in result:
        result["job_title"] = "Unknown Role"

    missing = {"resume_text", "jd_text", "label"} - set(result.columns)
    if missing:
        raise ValueError(
            f"Could not infer required columns {sorted(missing)} from dataset columns {list(frame.columns)}"
        )
    return result[STANDARD_COLUMNS]


def _expand_combined_hf_text(frame: pd.DataFrame) -> pd.DataFrame:
    """Split the real HF dataset's combined prompt into JD and resume columns."""
    if {"resume_text", "jd_text"}.issubset(frame.columns) or "text" not in frame.columns:
        return frame

    extracted = frame["text"].astype(str).map(_split_hf_prompt)
    result = frame.copy()
    result["jd_text"] = extracted.map(lambda item: item[0])
    result["resume_text"] = extracted.map(lambda item: item[1])
    result["job_title"] = "Hugging Face Resume-JD Match"
    result["candidate_name"] = [f"HF Candidate {index + 1}" for index in range(len(result))]
    return result


def _split_hf_prompt(text: str) -> tuple[str, str]:
    jd_start = text.find("<<")
    jd_end = text.find(">>", jd_start + 2)
    resume_marker_index = text.lower().find("the resume:", jd_end + 2)
    resume_start = text.find("<<", resume_marker_index)
    resume_end = text.rfind(">>")

    if min(jd_start, jd_end, resume_marker_index, resume_start, resume_end) < 0 or resume_start <= jd_end or resume_end <= resume_start:
        raise ValueError("Malformed dataset prompt: expected separate <<JD>> and the resume: <<resume>> sections")

    jd_text = text[jd_start + 2 : jd_end] if jd_start != -1 and jd_end != -1 else text
    if resume_start != -1 and resume_end != -1 and resume_end > resume_start:
        resume_text = text[resume_start + 2 : resume_end]
    elif resume_marker_index != -1:
        resume_text = text[resume_marker_index + len("the resume:") :]
    else:
        resume_text = text
    return jd_text, resume_text


def _normalize_label(value: object) -> str:
    text = str(value).strip().lower()
    if text in {"2", "true", "yes", "match", "matched", "good", "good fit", "high", "strong", "strong_fit"}:
        return "strong_fit"
    if text in {"1", "partial", "medium", "maybe", "potential", "potential fit", "potential_fit"}:
        return "potential_fit"
    if text in {"0", "false", "no", "no fit", "not_match", "low", "weak", "low_fit"}:
        return "low_fit"
    if text.replace(" ", "_") in {"strong_fit", "potential_fit", "low_fit"}:
        return text.replace(" ", "_")
    return text
