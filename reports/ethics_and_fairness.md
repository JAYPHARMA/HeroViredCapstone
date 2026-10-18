# Ethics, fairness and responsible use

This is decision support for recruiter review. Never use the score as the sole reason to shortlist or reject a candidate.
Explicit demographic/contact-field redaction runs before vectorization, embedding generation and skill extraction. Display names do not enter the model. Model exports exclude the document embedding cache.
Controlled tests vary explicit age, gender, religion, caste and marital-status fields while holding experience fixed. See `responsible_use_checks.json` for measured results.
These checks test the redaction path only. They do not establish demographic parity, equalized odds, or freedom from proxy bias.
Free-form identity mentions, education/geography proxies, source label bias, domain imbalance, keyword stuffing and negation remain risks. Do not infer protected groups from names or resume text.
Keep demographic outcome analysis as future work requiring appropriate consent and reliable labels. Do not manufacture group labels for this capstone.
Embedding skill suggestions require human confirmation and do not count as verified proficiency. Inspect source evidence and validate skill depth in interviews.
The local app does not deliberately write new resumes to disk. Process caches may retain normalized text until restart. Avoid real sensitive resumes in classroom demonstrations and restrict access to any deployed service.
For a future production system, establish consent, access controls, retention/deletion rules, calibrated uncertainty, role-specific error monitoring and recruiter correction workflows.
