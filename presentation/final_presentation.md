# Final eight-slide presentation · Sanket Toche

Editable deck: `output/HeroVired_Capstone_Sanket_Toche.pptx`. Speaker notes accompany every slide.

1. **Title and problem context.** Resume Screening and Job Description Matching. Recruiter decision support and explainable candidate comparison.
2. **Dataset source and access.** Pinned Hugging Face snapshot. 8,000 original pairs; 6228 clean source training pairs; 5025 development training, 1203 validation and 1759 original test pairs. Illustrative schema example.
3. **EDA and preprocessing.** Class distribution, conflicting labels, normalization, redaction, document lengths and chunking.
4. **Baseline and advanced architecture.** TF-IDF + Random Forest versus frozen Sentence-BERT paired embeddings + selected classifier. The encoder is not fine-tuned on capstone labels.
5. **Experiments and hyperparameters.** Validation comparison, regularization/leaf-size settings, selection by macro F1, learning curve and frozen encoder settings.
6. **Evaluation and error analysis.** Advanced accuracy 0.4792, macro F1 0.4489, NDCG@3 0.6910; native confusion matrix and latency evidence.
7. **Deployment demo.** Saved local model, three actual predictions, ranking CSV and keyword-only explanation failure.
8. **Risks and future work.** Human review, controlled redaction checks, uncertain source labels, shared resumes, unseen-candidate validation and recruiter feedback.

Measured test comparison:

| Model | Accuracy | Macro F1 | Weighted F1 | NDCG@3 | Precision@3 |
|---|---:|---:|---:|---:|---:|
| Majority | 0.4872 | 0.2184 | 0.3192 | 0.4983 | 0.2147 |
| Baseline | 0.4673 | 0.3626 | 0.4200 | 0.6150 | 0.2896 |
| Advanced | 0.4792 | 0.4489 | 0.4812 | 0.6910 | 0.3060 |
