# Problem Statement and Scope

## Business Context

Recruiters and placement teams review large numbers of resumes for each job description. Manual screening is slow, inconsistent, and difficult to explain. This project provides a decision-support prototype that ranks candidates, predicts fit, and highlights skill gaps.

## Objective

Build a resume-JD matching system that:

- Accepts a job description and multiple resumes.
- Ranks candidates by match score.
- Predicts fit category: `strong_fit`, `potential_fit`, or `low_fit`.
- Explains matched skills and missing skills.
- Provides a reproducible training, evaluation, and deployment workflow.

## Target Users

- Recruiters
- Career services teams
- Placement teams
- HR analysts
- Learners preparing for job applications

## In Scope

- Resume and job description text cleaning
- Hugging Face dataset access through `facehuggerapoorv/resume-jd-match`
- Semantic similarity baseline
- Fit classification
- Skill-gap extraction
- Streamlit dashboard
- FastAPI inference endpoint
- Evaluation report and responsible-use checklist

## Out of Scope

- Automated rejection or hiring decisions
- Protected attribute inference or use
- ATS production integration
- Claims of legal compliance
- PDF parsing at production quality

## Success Criteria

- Model can be trained reproducibly.
- Prototype loads a saved artifact for inference.
- Dashboard produces ranked candidates and explanations.
- Evaluation includes class-level metrics and a known failure case.
- Documentation explains setup, limitations, and ethical risks.
