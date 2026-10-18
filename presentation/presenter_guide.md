# Presenter guide · Sanket Toche

## Before presenting

1. Open `presentation/output/HeroVired_Capstone_Sanket_Toche.pptx` and read its speaker notes.
2. Double-click `launch_dashboard.cmd` from the project folder. Leave its terminal open.
3. Open http://localhost:8501 and wait for the three default predictions.
4. Keep `reports/model_evaluation.md`, the four notebooks and the dashboard available.
5. The prepared demo uses saved local weights. It needs no retraining or network connection.

## Eight-slide talk track · approximately 8–10 minutes

**Slide 1 · Problem, 40 seconds**

“I am Sanket Toche. This project helps recruiters compare multiple resumes with a job description. It predicts fit, ranks candidates, and explains matched and missing skills. Recruiters verify the evidence before making decisions.”

**Slide 2 · Dataset, 70 seconds**

“The source is the public Hugging Face resume–JD dataset with 8,000 pairs. I validate the format and labels, quarantine contradictory training pairs, and use validation JDs that do not occur in development training. The original test set remains separate. Most test resumes recur in training, so I claim evaluation on unseen JDs, not entirely unseen candidates.”

Use the actual split counts in the slide and run manifest. Explain the corrected dataset identifier if asked.

**Slide 3 · EDA, 60 seconds**

“Low-fit is the largest class. This is why I report macro F1 alongside accuracy. I checked missing content, duplicates, conflicting labels, text lengths and split overlap. Explicit demographic fields and selected boilerplate are removed. Long documents use chunks because the embedding model has a limited context window.”

**Slide 4 · Architecture, 80 seconds**

“The baseline combines TF-IDF cosine and overlap features with Random Forest. The advanced model uses a frozen pretrained MiniLM Sentence-BERT encoder. It produces a 384-dimensional representation for each document. I combine the resume vector, JD vector, their absolute difference and element-wise product, plus cosine and lexical features. A classifier learns the three fit labels. I trained the classifier, not the transformer weights.”

**Slide 5 · Experiments, 70 seconds**

“I compare embedding cosine, regularized logistic classifiers and tree ensembles. I choose the advanced configuration using validation macro F1. I record configurations, metrics and training time. The learning curve varies classifier training-set size on fixed validation JDs. Encoder learning rate and epochs do not apply because the encoder stays frozen.”

**Slide 6 · Evaluation, 90 seconds**

Read the measured values on the slide. Explain the majority baseline, class recall, ranking metrics and one false-negative example from the report.

“NDCG@3 rewards putting higher-fit candidates near the top. Precision@3 measures how many of the top three have strong-fit labels. Both average within JD groups. These labels and metrics are imperfect, so I do not claim hiring validity or statistical significance.”

**Slide 7 · Live demo, 90 seconds**

Show the existing three inputs. Click Rank candidates. Explain one matched skill and one missing skill. Expand prediction details and possible skill paraphrases if any appear. Download the CSV. Open Model evidence to show the comparison. Open Known failure case and run it.

“The keyword-only resume explicitly lacks experience, but the extractor still detects the mentioned tools. This shows the difference between a skill mention and verified proficiency.”

Use the actual outputs, even if an obvious example gets an imperfect category. Do not change labels manually to make the demonstration look better.

**Slide 8 · Responsible use and future work, 60 seconds**

“The system supports transparent comparison. It cannot replace recruiter verification. Controlled demographic-field checks verify that explicit fields are removed, but this does not prove fairness. Future work includes unseen-candidate testing, role-specific evaluation, recruiter-reviewed explanations, calibration and task-specific reranking.”

## Likely viva questions

**Did you fine-tune BERT?**

No. I use a pretrained Sentence-BERT encoder as a frozen feature extractor and train a supervised classifier on paired document features. This is transfer learning, not task-specific transformer fine-tuning.

**Why use TF-IDF and Sentence-BERT?**

TF-IDF gives a transparent lexical baseline. Sentence-BERT adds representations that can connect paraphrases. The same held-out test set allows a measured comparison.

**Why not tune with the test set?**

Repeated test-based choices make results optimistic. Configuration selection uses validation macro F1. The final model refits on clean source training before test evaluation.

**Why group by job description?**

Each JD appears in several pair rows. Random row splitting can expose the same JD during training and validation. Grouping tests transfer to new JDs.

**Why macro F1?**

Low-fit is the largest class. Macro F1 gives equal weight to all three classes and makes weak potential/strong-fit recognition visible. Accuracy and class-wise metrics remain in the report.

**Why is training performance much better than validation?**

The learning curve shows overfitting and unstable generalization from high-dimensional pair features and limited/uncertain labels. Validation controls model selection, but it cannot remove this gap. I report it as a limitation. Future experiments should compare stronger regularization, simpler pair features, more labels and grouped cross-validation before any production use.

**Is the score the probability a person should be hired?**

No. It is a composite review priority: 30% expected fit, 45% embedding cosine and 25% keyword overlap. The classifier label is a separate diagnostic and can disagree with ranking. Its probabilities are not calibrated hiring probabilities. Short inputs can produce unreliable categories, so the app shows warnings and retains the actual classifier output. The composite ranking gives document/skill evidence substantial weight.

**Does redaction prove fairness?**

No. The tests only establish invariance for a small set of explicit demographic-field variants. Free-form identity, proxy bias and biased source labels remain possible.

**Can it read PDF resumes?**

The demonstrated interface accepts pasted text. Production-quality PDF/OCR parsing is outside this implementation. The dataset parser handles the source's combined JD/resume text format.

**Why not claim the model understands experience depth?**

Neither keyword mentions nor document similarity prove proficiency. Negation, context, and unsupported claims still fail. Recruiters must inspect project evidence and validate experience.

**What is the strongest generalization limitation?**

Test JDs are unseen, but most test resumes appear elsewhere in training. A future split must evaluate entirely unseen candidates separately. Dataset provenance and label quality also remain uncertain.

**Did this save recruiters time?**

No recruiter timing study was performed. The prototype enables automated comparison and explanations, but measured business time savings are future validation work.

**Are Docker and cloud deployment verified?**

The local Streamlit and API are the validated deployment prototype. A Dockerfile is supplied as packaging guidance. Only claim a container/cloud run if it has actually been built and exercised.

## If the demo fails during the presentation

Use the saved `reports/demo_results.json`, held-out predictions, and slides. Restart with `launch_dashboard.cmd`. Use the project `.venv`, not the system Python. Keep the encoder directory with the project. No training is required to demonstrate inference.
