# Model Evaluation Report

Dataset: Hugging Face `facehuggerapoorv/resume-jd-match` test split when using the default trained artifact.

## Holdout Classification Metrics

```text
precision    recall  f1-score   support

      low_fit       0.51      0.76      0.61       857
potential_fit       0.33      0.18      0.23       444
   strong_fit       0.39      0.21      0.27       458

     accuracy                           0.47      1759
    macro avg       0.41      0.38      0.37      1759
 weighted avg       0.43      0.47      0.43      1759
```

## Confusion Matrix

Labels: low_fit, potential_fit, strong_fit

```text
[[653 110  94]
 [315  78  51]
 [318  46  94]]
```

## Sample Predictions

- HF Candidate 1: actual `low_fit`, predicted `strong_fit`, score `52.68`, missing skills: communication
- HF Candidate 2: actual `low_fit`, predicted `low_fit`, score `47.6`, missing skills: hr analytics, javascript, monitoring, rest api
- HF Candidate 3: actual `low_fit`, predicted `low_fit`, score `30.23`, missing skills: communication, javascript, rest api
- HF Candidate 4: actual `low_fit`, predicted `low_fit`, score `29.55`, missing skills: cloud deployment, communication, hr analytics, java, kubernetes
- HF Candidate 5: actual `low_fit`, predicted `low_fit`, score `48.4`, missing skills: excel, hr analytics, model evaluation
- HF Candidate 6: actual `low_fit`, predicted `low_fit`, score `27.56`, missing skills: analytics
- HF Candidate 7: actual `low_fit`, predicted `low_fit`, score `19.43`, missing skills: cloud deployment, communication
- HF Candidate 8: actual `low_fit`, predicted `strong_fit`, score `41.38`, missing skills: cloud deployment, feature engineering, java, kubernetes, monitoring
- HF Candidate 9: actual `low_fit`, predicted `low_fit`, score `34.32`, missing skills: cloud deployment, communication, hr analytics, testing
- HF Candidate 10: actual `low_fit`, predicted `low_fit`, score `25.49`, missing skills: analytics, cloud deployment, communication, docker, machine learning
- HF Candidate 11: actual `low_fit`, predicted `low_fit`, score `34.51`, missing skills: hr analytics
- HF Candidate 12: actual `low_fit`, predicted `low_fit`, score `30.49`, missing skills: ci/cd, cloud deployment

## Known Failure Case

Keyword-based skill extraction can over-credit resumes that list many tools without proving depth. The prototype should be used for recruiter decision support, not automated rejection.