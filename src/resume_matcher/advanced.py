"""Frozen Sentence-BERT document embeddings and a supervised fit classifier."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import re
import numpy as np

from .model import ResumeJDMatcher
from .preprocessing import clean_text, redact_personal_details
from .skills import SKILL_TAXONOMY, analyze_skill_gap

ROOT = Path(__file__).resolve().parents[2]
ENCODER_ID = "sentence-transformers/all-MiniLM-L6-v2"
ENCODER_DIR = ROOT / "models" / "encoder"


@lru_cache(maxsize=1)
def load_encoder():
    import torch
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(min(6, torch.get_num_threads()))
    if not (ENCODER_DIR / "modules.json").exists():
        raise FileNotFoundError("Local encoder missing. Run .venv/Scripts/python scripts/prepare_encoder.py once.")
    encoder = SentenceTransformer(str(ENCODER_DIR), device="cpu")
    encoder.max_seq_length = 256
    encoder.eval()
    return encoder


def document_chunks(text: str) -> list[str]:
    """Cover the document with up to 8 evenly spaced 150-word chunks."""
    words = clean_text(text).split()
    if not words:
        raise ValueError("Text contains no usable content after preprocessing")
    starts = list(range(0, len(words), 150))
    if len(starts) > 8:
        starts = [starts[i] for i in np.linspace(0, len(starts)-1, 8, dtype=int)]
    return [" ".join(words[start:start+150]) for start in starts]


def encode_documents(texts: list[str], show_progress: bool = False) -> np.ndarray:
    chunks, owners = [], []
    for index, text in enumerate(texts):
        parts = document_chunks(text)
        chunks.extend(parts)
        owners.extend([index]*len(parts))
    embeddings = load_encoder().encode(chunks, batch_size=32, normalize_embeddings=True,
                                       show_progress_bar=show_progress, convert_to_numpy=True)
    result = np.zeros((len(texts), embeddings.shape[1]), dtype=np.float32)
    for owner, embedding in zip(owners, embeddings):
        result[owner] += embedding
    result /= np.maximum(np.linalg.norm(result, axis=1, keepdims=True), 1e-9)
    return result


class EmbeddingResumeJDMatcher(ResumeJDMatcher):
    """384-D chunk-pooled embeddings, paired features, and a selected classifier.

    The encoder stays frozen. Only the sklearn classifier is trained on our data.
    """
    def __init__(self, classifier=None):
        super().__init__()
        if classifier is not None:
            self.classifier = classifier
        self.embedding_cache = {}
        self.metadata = {}

    def embed(self, texts):
        keys = [clean_text(t) for t in texts]
        if len(self.embedding_cache) > 4096:
            self.embedding_cache = {}
        missing = list(dict.fromkeys(k for k in keys if k not in self.embedding_cache))
        if missing:
            values = encode_documents(missing)
            self.embedding_cache.update(zip(missing, values))
        return np.stack([self.embedding_cache[k] for k in keys])

    def _feature_matrix(self, resumes, jds):
        resumes, jds = list(resumes), list(jds)
        r, j = self.embed(resumes), self.embed(jds)
        lexical = super()._feature_matrix(resumes, jds)
        cosine = np.sum(r*j, axis=1, keepdims=True)
        # Cosine first so legacy inference exposes actual embedding similarity.
        return np.hstack([cosine, lexical, r, j, np.abs(r-j), r*j]).astype(np.float32)

    def predict_one(self, resume_text, jd_text, candidate_name="Candidate"):
        if not clean_text(resume_text) or not clean_text(jd_text):
            raise ValueError("Provide non-empty resume and job description text")
        item = super().predict_one(resume_text, jd_text, candidate_name)
        # Rank by combined evidence. Preserve the trained classifier's actual label
        # as a separate diagnostic instead of forcing ordering from a coarse label.
        expected = (item["probabilities"].get("potential_fit", 0) + 2*item["probabilities"].get("strong_fit", 0))/2
        score = .30*expected + .45*np.clip(item["semantic_similarity"],0,1) + .25*item["skill_overlap_ratio"]
        item["match_score"] = round(float(score)*100, 2)
        item["score_definition"] = "Composite priority: 30% expected fit, 45% embedding cosine, 25% keyword overlap. Classifier label is separate and may disagree. Not a hiring probability."
        item["model_name"] = self.metadata.get("selected_model", "Sentence-BERT paired classifier")
        item["semantic_skill_suggestions"] = self.semantic_skill_suggestions(resume_text, jd_text)
        item["input_warnings"] = []
        if min(len(clean_text(resume_text).split()),len(clean_text(jd_text).split())) < 80:
            item["input_warnings"].append("Short inputs differ from many source examples; the category and probabilities can be overconfident. Verify the full resume.")
        if item["predicted_fit"] != "low_fit" and item["semantic_similarity"] < .2:
            item["input_warnings"].append("Predicted fit conflicts with low document similarity. Manual review is required.")
        return item

    def semantic_skill_suggestions(self, resume_text, jd_text):
        """Flag paraphrases for human review; never silently mark skills as proved."""
        missing = analyze_skill_gap(resume_text, jd_text).missing_skills
        if not missing:
            return []
        sentences = [s.strip() for s in re.split(r"[\n.!?]+", redact_personal_details(resume_text)) if len(s.strip().split()) >= 4][:32]
        if not sentences:
            return []
        labels = ["Experience in " + skill + ": " + ", ".join(SKILL_TAXONOMY[skill]) for skill in missing]
        scores = self.embed(labels) @ self.embed(sentences).T
        suggestions = []
        for i, skill in enumerate(missing):
            best = int(scores[i].argmax())
            if scores[i, best] >= .42:
                suggestions.append({"skill": skill, "similarity": round(float(scores[i,best]),3),
                                    "evidence": sentences[best][:240], "status": "Unverified paraphrase; recruiter review required"})
        return suggestions

    def save(self, path):
        # Do not serialize training resume text/embeddings into the deployable artifact.
        cache = self.embedding_cache
        self.embedding_cache = {}
        try:
            super().save(path)
        finally:
            self.embedding_cache = cache
