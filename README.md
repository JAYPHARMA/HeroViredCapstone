# Resume Screening and Job Description Matching

HeroVired AI/ML Specialisation Capstone, Project 5. Presenter: **Sanket Toche**.

Ranks resumes against a JD, predicts low/potential/strong fit, and explains keyword gaps plus possible semantic paraphrases. Recruiters must verify experience and retain every hiring decision.

## Clone from GitHub

Install Git LFS before cloning. The model weights, reproducibility CSVs and submission ZIP use LFS, so a source-only download may contain pointer files instead of those artifacts.

```powershell
git lfs install
git clone https://github.com/JAYPHARMA/HeroViredCapstone.git
cd HeroViredCapstone
git lfs pull
```

Then follow **Set up on another computer** below. The eight-slide deck is in `presentation/output/`; the complete portable package is in `submission/`.

## Present the prepared project

The trained classifier and local Sentence-BERT encoder are already included. No retraining or model download is needed for this machine's demo.

```powershell
cd A:\HEROVIRED
.\.venv\Scripts\python.exe -m streamlit run app_streamlit.py
```

Or double-click **launch_dashboard.cmd**. Open http://localhost:8501. The default JD and three candidate examples are ready.

1. Show candidate rankings, keyword gaps and prediction details.
2. Download the CSV.
3. Open **Model evidence** for test metrics and plots.
4. Open **Known failure case** and demonstrate why skill mentions do not establish proficiency.

Presentation: `presentation/output/HeroVired_Capstone_Sanket_Toche.pptx`. Presenter script, demo checklist and viva questions: `presentation/presenter_guide.md`.

## Run the API

```powershell
.\.venv\Scripts\python.exe -m uvicorn api_fastapi:app --host 127.0.0.1 --port 8000
```

Or double-click **launch_api.cmd**. Interactive documentation: http://127.0.0.1:8000/docs. Health: http://127.0.0.1:8000/health.

```json
{
  "jd_text": "Python engineer with SQL and Docker experience required.",
  "candidates": [
    {"candidate_name": "Example A", "resume_text": "Python engineer building SQL services and Docker containers."},
    {"candidate_name": "Example B", "resume_text": "Graphic designer specializing in print layouts and illustration."}
  ]
}
```

POST this body to `/match`. Inputs must contain 20–50,000 characters each, with 1–50 candidates. Blank inputs are rejected. Paths resolve from the project directory, independent of the shell's current working directory.

## Set up on another computer

Use **Python 3.11**. Keep `models/encoder/` and `models/resume_matcher.joblib` with the project for offline inference.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m pip install --no-deps -e .
.\.venv\Scripts\python.exe -m streamlit run app_streamlit.py
```

The constraints pin measured direct dependency versions, including compatible Torch, Transformers and Sentence Transformers releases. They are not a complete transitive/hash lock. `reports/environment.json` records the run environment. Do not use an unrelated system Python to launch the advanced model.

## Reproduce the research

```powershell
# Only needed if encoder weights are missing; downloads and records a revision.
.\.venv\Scripts\python.exe scripts/prepare_encoder.py

# Audits archived public data, creates grouped validation, runs configurations,
# selects on validation macro F1, refits, evaluates, and records artifacts/hashes.
.\.venv\Scripts\python.exe scripts/run_capstone.py

# Recheck test metrics using the deployed score precision, with no fitting.
.\.venv\Scripts\python.exe scripts/recheck_evaluation.py

# Optional standalone latency measurement; run before rebuilding reports.
.\.venv\Scripts\python.exe scripts/benchmark.py

# Build measured reports, plots, example outputs and notebook sources.
.\.venv\Scripts\python.exe scripts/build_submission.py
.\.venv\Scripts\python.exe scripts/execute_notebooks.py

# Regression checks for the model, data protocol and API.
.\.venv\Scripts\python.exe -m pytest tests -q
```

Optional `run_capstone.py --download` reloads dataset revision `bfab7acecac062628b39e0dd0c59e72d675edc9c` from `facehuggerapoorv/resume-jd-match`. The brief's `resumejd-match` identifier differs from the accessible repository; the corrected identifier is intentional.

The full runner replaces the deployed capstone artifact. The older `scripts/train_model.py` command instead writes a separate **sample_baseline.joblib** and separate sample splits, so its 12-row demo dataset cannot accidentally replace the presentation model through the default command.

## Method and evaluation

- **Baseline:** TF-IDF cosine, skill/token overlap and lengths, then a 250-tree Random Forest.
- **Semantic baseline:** frozen Sentence-BERT cosine with a logistic fit classifier, recorded in validation experiments.
- **Advanced:** frozen `all-MiniLM-L6-v2`, up to eight 150-word chunks per document, pooled 384-D embeddings. Concatenate resume/JD embeddings, absolute difference, element-wise product, cosine and lexical features (1,545 dimensions), then train a supervised classifier.
- **Selection:** compare logistic regularization and tree-ensemble settings on JD-grouped validation macro F1. Do not select configurations from test results. The encoder is not fine-tuned on capstone labels.
- **Metrics:** accuracy, macro/weighted/class-wise F1, confusion matrices, JD-grouped NDCG@3, strong-fit Precision@3, sample errors and CPU latency. See `reports/model_evaluation.md` for measured values and definitions.
- **Score:** a composite review priority: 30% expected fit, 45% embedding cosine and 25% keyword overlap. Classifier categories remain separate predictions and can disagree with rank; the UI explains both rather than relabeling candidates to match a score. It is not a calibrated hiring probability. Short inputs and category/similarity contradictions show warnings.

## Submission files

```text
notebooks/01_data_access_preprocessing.ipynb
notebooks/02_eda.ipynb
notebooks/03_baseline_model.ipynb
notebooks/04_advanced_model.ipynb
reports/problem_scope.md
reports/data_quality.md
reports/model_evaluation.md
reports/ethics_and_fairness.md
reports/requirements_alignment.md
reports/figures/
reports/experiments/
reports/run_manifest.json
reports/responsible_use_checks.json
presentation/output/
presentation/presenter_guide.md
```

Training documents/embeddings are excluded from the deployed classifier artifact. The optional research cache `models/document_embeddings.joblib` and raw/processed CSVs exist for notebook and experiment reproduction, not inference.

## Limitations and responsible use

The upstream dataset card is empty, so annotation provenance and synthetic-data status remain uncertain. Contradictory training pairs are quarantined. Test JDs are unseen, but most test resumes recur in training. Unseen-candidate performance remains unproven.

Explicit demographic/contact-field redaction and controlled invariance checks reduce direct exposure. They do not prove demographic fairness. Free-form identity, proxy bias, negation, unsupported skill claims and domain gaps remain. Semantic skill suggestions require recruiter confirmation.

The local app accepts pasted text. Production PDF/OCR, ATS integration, user authentication, calibrated hiring decisions, public hosting and proven recruiter time savings are outside the validated scope. The local app does not deliberately save new input resumes to disk, but in-process caches retain normalized text until restart.

## Optional container packaging

```powershell
docker build -t herovired-matcher .
docker run --rm -p 8501:8501 herovired-matcher
```

Docker packaging is supplied as an optional deployment path. A Docker/cloud run is not claimed unless recorded separately; the tested prototype is local Streamlit and FastAPI.
