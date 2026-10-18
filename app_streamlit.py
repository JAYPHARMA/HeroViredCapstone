from __future__ import annotations
import json
import re
from pathlib import Path
import pandas as pd
import streamlit as st
from resume_matcher.model import ResumeJDMatcher

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="Resume-JD Matcher", page_icon="📄", layout="wide")


@st.cache_resource
def load_model():
    return ResumeJDMatcher.load(ROOT / "models/resume_matcher.joblib")


def split_resumes(raw_text):
    blocks = [b.strip() for b in re.split(r"(?m)^\s*---\s*$", raw_text) if b.strip()]
    return [
        {"candidate_name": f"Candidate {i+1}", "resume_text": b}
        for i, b in enumerate(blocks)
    ]


st.title("Resume screening and JD matching")
st.caption("HeroVired AI/ML Capstone · Sanket Toche")
st.info(
    "Recruiter decision support. Review the original resume and evidence before making a decision."
)
demo = json.loads((ROOT / "data/demo_cases.json").read_text(encoding="utf-8"))
try:
    model = load_model()
except Exception as exc:
    st.error(
        f"Could not load the model: {exc}. Check that the saved model and encoder files were downloaded with Git LFS."
    )
    st.stop()

tab_match, tab_evidence, tab_failure = st.tabs(
    ["Match candidates", "Model evidence", "Known failure case"]
)
with st.sidebar:
    st.header("Job and resumes")
    with st.form("matching_inputs"):
        jd = st.text_area(
            "Job description", demo["jd_text"], height=200, max_chars=50000
        )
        raw = st.text_area(
            "Candidate resumes",
            "\n---\n".join(c["resume_text"] for c in demo["candidates"]),
            height=330,
            help="Separate resumes with --- on a line by itself.",
        )
        submitted = st.form_submit_button("Rank candidates", type="primary")
    st.caption(
        "Plain text inputs. Candidate names are display labels and do not enter the model."
    )

if submitted or "ranked" not in st.session_state:
    candidates = split_resumes(raw)
    if (
        len(jd.strip()) < 20
        or not candidates
        or any(
            len(c["resume_text"]) < 20 or len(c["resume_text"]) > 50000
            for c in candidates
        )
        or len(candidates) > 50
    ):
        st.session_state.ranked = []
        st.error("Enter a JD and 1–50 resumes, each containing 20–50,000 characters.")
    else:
        with st.spinner("Comparing experience and skills…"):
            try:
                st.session_state.ranked = model.rank_candidates(jd, candidates)
            except ValueError as exc:
                st.error(str(exc))

with tab_match:
    st.subheader("Candidate ranking")
    st.caption(
        "Ranking blends 30% expected fit, 45% embedding similarity and 25% keyword overlap. The classifier category is a separate prediction and can disagree. Scores are not hiring probabilities."
    )
    ranked = st.session_state.get("ranked", [])
    for index, item in enumerate(ranked, 1):
        with st.container(border=True):
            c1, c2, c3 = st.columns([1, 2, 5])
            c1.metric("Rank", index)
            c2.metric("Review priority", f"{item['match_score']:.1f} / 100")
            c3.markdown(
                f"**{item['candidate_name']} · {item['predicted_fit'].replace('_',' ').title()}**"
            )
            c3.write(item["recommendation"])
            for warning in item.get("input_warnings", []):
                st.warning(warning)
            st.write(
                "Matched keywords:",
                ", ".join(item["skill_gap"]["matched_skills"]) or "None detected",
            )
            st.write(
                "Missing keywords:",
                ", ".join(item["skill_gap"]["missing_skills"]) or "None detected",
            )
            suggestions = item.get("semantic_skill_suggestions", [])
            if suggestions:
                with st.expander("Possible skill paraphrases requiring review"):
                    for suggestion in suggestions:
                        st.write(
                            f"{suggestion['skill']} (similarity {suggestion['similarity']}): {suggestion['evidence']}"
                        )
                    st.caption(
                        "Semantic similarity does not prove experience. Suggestions do not automatically change the keyword skill list."
                    )
            with st.expander("Prediction details"):
                st.json(
                    {
                        "fit_probabilities_uncalibrated": item["probabilities"],
                        "embedding_cosine": item["semantic_similarity"],
                    }
                )
    table = pd.DataFrame(
        [
            {
                "candidate": r["candidate_name"],
                "fit": r["predicted_fit"],
                "priority_score": r["match_score"],
                "missing_skills": ", ".join(r["skill_gap"]["missing_skills"]),
            }
            for r in ranked
        ]
    )
    st.dataframe(table, hide_index=True, width="stretch")
    st.download_button(
        "Download rankings CSV",
        table.to_csv(index=False),
        "candidate_rankings.csv",
        "text/csv",
    )

with tab_evidence:
    metrics_path = ROOT / "reports/metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())
        st.subheader("Held-out test results")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "model": name,
                        "accuracy": m["accuracy"],
                        "macro_F1": m["macro_f1"],
                        "NDCG@3": m["ranking"]["ndcg_at_3"],
                        "Precision@3": m["ranking"]["precision_at_3_strong_fit"],
                    }
                    for name, m in metrics.items()
                ]
            ),
            hide_index=True,
        )
        st.caption(
            "Configuration selected on a validation split grouped by JD. Test JDs are unseen, but most resumes also occur in training. Precision@3 treats strong-fit labels as relevant."
        )
        for name in ["confusion_matrix", "model_comparison", "learning_curve"]:
            image_path = ROOT / f"reports/figures/{name}.png"
            if image_path.exists():
                st.image(str(image_path))
    st.write(
        "Architecture: frozen MiniLM Sentence-BERT encoder, document chunks, paired embeddings, lexical/skill features, supervised classifier."
    )
    st.write(
        "Explicit demographic/contact-field redaction reduces direct exposure. Free-form sensitive text and proxy bias may remain."
    )

with tab_failure:
    st.subheader("Keyword lists can overstate experience")
    st.write(demo["failure_case"]["explanation"])
    st.code(demo["failure_case"]["resume_text"])
    if st.button("Run known failure example"):
        with st.spinner("Running failure example…"):
            st.json(
                model.predict_one(
                    demo["failure_case"]["resume_text"],
                    demo["jd_text"],
                    "Keyword-only example",
                )
            )
    st.warning(
        "Negation, experience depth, and unsupported skill claims still require human review. No automated rejection."
    )
