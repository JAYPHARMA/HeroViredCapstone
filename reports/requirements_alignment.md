# HeroVired requirement alignment

Presenter: **Sanket Toche**. This map compares the supplied Project 5 brief against the delivered local project.

Verification evidence: 15 passing regression tests (`test_results.xml`), four notebooks with executed outputs and no error outputs, and eight slides in the editable deck.

| Brief item | Evidence and status |
|---|---|
| Sections 1–4: context, objective, users and scope | Implemented in problem_scope.md and slides 1–2. |
| Sections 5–6: public dataset access and preprocessing | Pinned source revision, strict parsing, schema/label mapping, normalization, selected boilerplate removal, audits, CSV hashes and notebook 01. |
| Section 6: sentence embeddings | Frozen MiniLM Sentence-BERT, chunk pooling and a recorded encoder revision. |
| Section 7: similarity baseline and fit classifier | TF-IDF/Random Forest baseline, embedding-cosine validation baseline and advanced paired-embedding classifier. |
| Section 7: skill-gap insights | Keyword gaps plus embedding-based paraphrase suggestions labeled unverified. Proficiency is not established from a mention. |
| Sections 7 and 10: multi-resume dashboard | Local Streamlit, saved model loading, three default examples, downloadable ranking CSV, model-evidence and known-failure tabs. |
| Section 8: validation and EDA | Audited contradictory pairs and duplicates, JD-grouped validation, notebook 02, class/text-length figures and leakage checks. |
| Section 8: experimentation and tuning | Recorded logistic regularization and ensemble parameters, validation metrics, classifier training times and learning curves. No encoder fine-tuning or invented optimizer/epoch experiment. |
| Section 9: evaluation | Accuracy, macro/weighted/class-wise F1, native/table and plotted confusion matrix, grouped NDCG@3 and strong-fit Precision@3, actual errors and CPU latency. |
| Section 9: explanation quality and fairness | Three keyword explanation fixtures, twelve explicit demographic-field variants across three resumes, a known negation/experience-depth failure. Small controlled checks only; no demographic fairness certification. |
| Section 10: model artifact and input/output | Saved classifier and local encoder; API validation, offline smoke checks and model serialization regression. |
| Section 10: sample predictions/failure and README | Three measured new-input predictions, reproducible keyword-only failure, run instructions and presenter guide. |
| Section 11: suggested team structure | Sanket Toche confirmed as presenter. Responsibility map supplied; additional team names/contribution claims need verified human input if required. |
| Section 12: problem/scope and data script | problem_scope.md, data_quality.md, data.py, run_capstone.py and notebook 01. |
| Section 12: EDA notebook | Executed notebook 02 with real outputs and plots. |
| Section 12: baseline and advanced notebooks | Executed notebooks 03–04, including actual classifier training and validation outputs. |
| Section 12: evaluation report | model_evaluation.md plus metrics, predictions, figures, checks, environment and run manifest. |
| Section 12: deployment prototype | Local Streamlit and FastAPI. Dockerfile supplied but container/cloud execution is not claimed. |
| Sections 12 and 15: final deck | Eight-slide editable PowerPoint with Sanket Toche, dataset, EDA, architecture, experiments, measured results, demo and limitations. Speaker notes and viva guide included. |
| Section 12: GitHub/project folder | Organized local project and portable ZIP with source/artifact hashes. Remote publication is not required for this folder-based submission and is not claimed. |
| Section 13: improved advanced model | Observed improvement: test macro F1 change versus the newly refitted baseline is +0.0863. These are point estimates, not a significance claim. |
| Sections 13–14: business interpretation and risks | Decision support, human verification, source uncertainty, repeated resumes, class errors, proxy bias and limits stated explicitly. No claimed hiring-validity or recruiter-time study. |

## Scope distinctions

The brief does not require implementing every listed CNN/RNN/transformer/generative architecture. The advanced path uses a pretrained transformer feature extractor and a trained classifier, with a justified frozen-encoder approach. It does not claim task-specific transformer fine-tuning or TensorFlow/Keras training.

Production ATS integration and automated hiring are explicitly out of scope. Production PDF/OCR, cloud hosting, authentication and Docker runtime validation are not claimed as completed work. The demonstrated interface accepts pasted text.

## Remaining limitations for the presentation

Source label provenance and synthetic-data status remain uncertain because the upstream card is empty. Most test resumes recur in training, despite unseen test JDs and disjoint exact pairs. Additional unseen-candidate, role-specific and recruiter-reviewed evaluation is future work. The small controlled redaction/explanation fixtures do not prove demographic fairness or semantic explanation accuracy.
