# Resume Screening and Job Description Matching

**Sanket Toche · HeroVired AI/ML Specialisation · Project 5**

This project compares resumes with a job description, ranks candidates and identifies skill gaps. It combines Sentence-BERT embeddings with a supervised fit classifier and a keyword-based explanation layer. A Streamlit dashboard and FastAPI endpoint provide two ways to use the saved model.

## Results

Evaluation uses 1,759 held-out resume–JD pairs. Model selection uses a separate validation split grouped by job description.

| Model | Accuracy | Macro F1 | NDCG@3 |
|---|---:|---:|---:|
| TF-IDF + Random Forest | 0.4673 | 0.3626 | 0.6150 |
| Sentence-BERT + Logistic Regression | 0.4792 | 0.4489 | 0.6910 |

The advanced model improves macro F1, although accuracy remains below the majority-class baseline (48.72%). See [the evaluation report](reports/model_evaluation.md) for class-level metrics, errors and ranking definitions.

## Installation

Requirements: Python 3.11, Git and Git LFS. Model weights, dataset CSVs and the submission ZIP use LFS.

```powershell
git lfs install
git clone https://github.com/JAYPHARMA/HeroViredCapstone.git
cd HeroViredCapstone
git lfs pull
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m pip install --no-deps -e .
```

The dependency file pins the direct package versions used for evaluation. The complete environment summary is in `reports/environment.json`.

## Dashboard

```powershell
.\.venv\Scripts\python.exe -m streamlit run app_streamlit.py
```

Open http://localhost:8501. On Windows, `launch_dashboard.cmd` runs the same command.

Hosted dashboard: [HeroVired Capstone](https://heroviredcapstone-galkgzfeasfbamryw8n92j.streamlit.app/).

Paste one job description and separate resumes with `---` on a line by itself. The dashboard displays rankings, fit categories, matched/missing keywords and possible skill paraphrases. Results can be downloaded as CSV. The sidebar menu also provides **How it works**, **Model evidence**, and **Known failure case**.

**How it works** contains interactive prediction and training diagrams for the presentation. Click components for source references, animate the flow, or download the standalone HTML. Open this section directly by adding `?page=how-it-works` to the app URL. The diagrams are bundled in `presentation/diagrams/` and need no separate hosting service.

The included model and local encoder support offline inference. Retraining is optional.

## Streamlit Community Cloud

On the deployment form, use:

| Field | Value |
|---|---|
| Repository | `JAYPHARMA/HeroViredCapstone` |
| Branch | `main` |
| Main file path | `app_streamlit.py` |
| Python version in Advanced settings | `3.11` |

Alternatively, paste `https://github.com/JAYPHARMA/HeroViredCapstone/blob/main/app_streamlit.py` into the GitHub URL field. Choose an available app subdomain and click Deploy. No API keys or secrets are required for this app.

`requirements.txt` installs the application package and runtime dependencies, including CPU-only PyTorch on Linux. `requirements-lock.txt` also includes the tools needed to reproduce the research locally. Community Cloud supports the repository's Git LFS model files.

## API

```powershell
.\.venv\Scripts\python.exe -m uvicorn api_fastapi:app --host 127.0.0.1 --port 8000
```

Interactive API documentation: http://127.0.0.1:8000/docs. `GET /health` checks readiness. `POST /match` accepts `jd_text` and a `candidates` list containing `candidate_name` and `resume_text`. Each text must contain 20–50,000 characters; a request supports 1–50 candidates.

## Method

The baseline uses TF-IDF cosine similarity, skill/token overlap and relative text length with a 250-tree Random Forest.

The advanced model uses the pretrained `all-MiniLM-L6-v2` encoder. Up to eight 150-word chunks represent each document as a pooled, normalized 384-dimensional vector. Resume and JD vectors, their absolute difference, element-wise product and overlap features form a 1,545-dimensional input to the classifier. The encoder stays frozen; the classifier learns the three fit labels.

Configurations vary logistic regularization and tree-ensemble parameters. Validation macro F1 selects the final classifier, which is then refitted on 6,228 cleaned source-training pairs.

Ranking combines 30% expected fit, 45% embedding cosine and 25% keyword overlap. The classifier category is a separate prediction and can disagree with rank. Neither output is a calibrated hiring probability.

## Reproducing the experiments

```powershell
.\.venv\Scripts\python.exe scripts/run_capstone.py
.\.venv\Scripts\python.exe scripts/recheck_evaluation.py
.\.venv\Scripts\python.exe scripts/benchmark.py
.\.venv\Scripts\python.exe scripts/build_submission.py
.\.venv\Scripts\python.exe scripts/execute_notebooks.py
.\.venv\Scripts\python.exe -m pytest tests -q --junitxml=reports/test_results.xml
```

`run_capstone.py` uses the archived public data. Add `--download` to reload its pinned Hugging Face revision. `prepare_encoder.py` downloads the encoder only if the local copy is missing. `train_model.py` is a separate 12-row sample exercise and writes `sample_baseline.joblib`.

## Project files

| Folder | Contents |
|---|---|
| `src/resume_matcher/` | Preprocessing, skill extraction, models and evaluation utilities |
| `scripts/` | Training, evaluation, benchmarking and packaging |
| `notebooks/` | Four executed notebooks: data access, EDA, baseline and advanced model |
| `reports/` | Metrics, experiment records, plots, error analysis and data audit |
| `models/` | Saved classifiers, local encoder and research embedding cache |
| `presentation/` | Eight-slide PowerPoint, speaker notes and presentation guide |
| `submission/` | Portable project ZIP |
| `tests/` | 16 regression checks covering preprocessing, inference, API and dashboard navigation |

## Limitations

- Test job descriptions are unseen, but most test resumes also occur in training. Performance on entirely new candidates needs a separate evaluation.
- Source annotation quality and synthetic-data status are uncertain. Contradictory training pairs are quarantined.
- Overfitting, short inputs, negation and unsupported skill lists can produce incorrect or overconfident categories.
- Explicit demographic/contact-field redaction passes controlled checks. Free-form identity and proxy bias can remain.
- The interface accepts pasted text. PDF/OCR, ATS integration and authentication are future work. The dashboard is hosted on Streamlit Community Cloud. The Dockerfile is available for packaging, but container execution has not been verified.

Recruiters should inspect the original resume and validate experience. Scores must not determine hiring or rejection on their own. Input resumes stay in process memory and are not deliberately written to disk by the local app.

## Data and model sources

- Dataset: [facehuggerapoorv/resume-jd-match](https://huggingface.co/datasets/facehuggerapoorv/resume-jd-match), revision `bfab7acecac062628b39e0dd0c59e72d675edc9c`.
- Encoder: [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2). Its model card and attribution remain in `models/encoder/README.md`.
- Reproducibility records: `reports/run_manifest.json`, `reports/dataset_source_verification.json` and `reports/submission_manifest.json`.
