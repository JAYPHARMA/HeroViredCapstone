from __future__ import annotations
import json
import re
from pathlib import Path
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
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


def ready_model():
    try:
        return load_model()
    except Exception as exc:
        st.error(
            f"Could not load the model: {exc}. Check that the saved model and encoder files were downloaded with Git LFS."
        )
        st.stop()


def show_matching(demo):
    model = ready_model()
    with st.sidebar:
        st.header("Job and resumes")
        with st.form("matching_inputs"):
            jd = st.text_area(
                "Job description",
                st.session_state.get("draft_jd", demo["jd_text"]),
                height=200,
                max_chars=50000,
            )
            raw = st.text_area(
                "Candidate resumes",
                st.session_state.get(
                    "draft_resumes",
                    "\n---\n".join(c["resume_text"] for c in demo["candidates"]),
                ),
                height=330,
                help="Separate resumes with --- on a line by itself.",
            )
            submitted = st.form_submit_button("Rank candidates", type="primary")
        st.caption(
            "Plain text inputs. Candidate names are display labels and do not enter the model."
        )

    st.session_state.draft_jd = jd
    st.session_state.draft_resumes = raw
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
            st.error(
                "Enter a JD and 1–50 resumes, each containing 20–50,000 characters."
            )
        else:
            with st.spinner("Comparing experience and skills…"):
                try:
                    st.session_state.ranked = model.rank_candidates(jd, candidates)
                except ValueError as exc:
                    st.error(str(exc))

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


def show_evidence():
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


def show_failure(demo):
    st.subheader("Keyword lists can overstate experience")
    st.write(demo["failure_case"]["explanation"])
    st.code(demo["failure_case"]["resume_text"])
    if st.button("Run known failure example"):
        model = ready_model()
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


def show_architecture():
    st.subheader("How it works")
    st.write(
        "Explore how resumes become ranked results, then see how the model was trained and evaluated."
    )
    view = st.radio(
        "Diagram",
        ["Prediction workflow", "Training and evaluation"],
        horizontal=True,
        key="architecture_view",
    )
    filename = (
        "prediction-workflow.html"
        if view == "Prediction workflow"
        else "training-evaluation.html"
    )
    diagram = ROOT / "presentation" / "diagrams" / filename
    if not diagram.exists():
        st.error(
            "The diagram file is missing. Check that presentation/diagrams was included in deployment."
        )
        return
    html = diagram.read_text(encoding="utf-8")
    st.caption(
        "Click a box to inspect its details and source code. Use Live to animate the flow, or Export to save an image."
    )
    st.download_button(
        "Download interactive diagram",
        data=html,
        file_name=filename,
        mime="text/html",
    )
    # Only embed the checked-in presentation asset, never uploaded/user HTML.
    components.html(html, height=1050, scrolling=True, tab_index=0)


st.title("Resume screening and JD matching")
st.caption("HeroVired AI/ML Capstone · Sanket Toche")
st.info(
    "Recruiter decision support. Review the original resume and evidence before making a decision."
)
SECTIONS = {
    "Match candidates": "match-candidates",
    "How it works": "how-it-works",
    "Model evidence": "model-evidence",
    "Known failure case": "known-failure-case",
}
requested = st.query_params.get("page", "match-candidates")
initial = next(
    (name for name, slug in SECTIONS.items() if slug == requested), "Match candidates"
)
section = st.sidebar.radio(
    "Menu", list(SECTIONS), index=list(SECTIONS).index(initial), key="section"
)
if st.query_params.get("page") != SECTIONS[section]:
    st.query_params["page"] = SECTIONS[section]
st.sidebar.caption("Sanket Toche · HeroVired AI/ML Capstone")

if section == "How it works":
    show_architecture()
elif section == "Model evidence":
    show_evidence()
else:
    demo = json.loads((ROOT / "data/demo_cases.json").read_text(encoding="utf-8"))
    if section == "Known failure case":
        show_failure(demo)
    else:
        show_matching(demo)
