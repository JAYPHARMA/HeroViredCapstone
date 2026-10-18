"""Build reports, evidence plots and executable notebooks from measured outputs."""
from __future__ import annotations
import json, textwrap
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nbformat as nbf
from resume_matcher.model import ResumeJDMatcher, LABEL_ORDER
from resume_matcher.preprocessing import clean_text
from resume_matcher.skills import analyze_skill_gap
from resume_matcher.research import ROOT, write_json

def read(path): return json.loads((ROOT/path).read_text(encoding='utf-8'))
def write(path,text):
    # Interpolated tables may contain unindented continuation lines.
    lines=text.splitlines()
    normalized='\n'.join(line[4:] if line.startswith('    ') else line for line in lines)
    (ROOT/path).write_text(normalized.strip()+'\n',encoding='utf-8')

def plot_confusion(matrix,title,path):
    fig,ax=plt.subplots(figsize=(7.6,5.2))
    ax.imshow(matrix,cmap='Blues')
    for i,row in enumerate(matrix):
        for j,value in enumerate(row): ax.text(j,i,str(value),ha='center',va='center',fontsize=14,color='white' if value>np.max(matrix)*.6 else '#14243A')
    ax.set(xticks=range(3),yticks=range(3),xticklabels=['Low','Potential','Strong'],yticklabels=['Low','Potential','Strong'],xlabel='Predicted fit',ylabel='Actual fit',title=title)
    fig.tight_layout(); fig.savefig(path,dpi=160); plt.close(fig)

def notebook(name,cells):
    nb=nbf.v4.new_notebook(cells=[nbf.v4.new_markdown_cell(textwrap.dedent(s).strip()) if kind=='md' else nbf.v4.new_code_cell(textwrap.dedent(s).strip()) for kind,s in cells])
    nb.metadata={'kernelspec':{'display_name':'HeroVired (.venv)','language':'python','name':'herovired'},'language_info':{'name':'python','version':'3.11'}}
    nbf.write(nb,ROOT/'notebooks'/name)

BOOT="""
from pathlib import Path
import sys, json
ROOT=Path.cwd() if (Path.cwd()/'src').exists() else Path.cwd().parent
sys.path.insert(0,str(ROOT/'src'))
import pandas as pd
import numpy as np
def read_json(path): return json.loads((ROOT/path).read_text())
"""

def main():
    metrics=read('reports/metrics.json'); audit=read('reports/data_audit.json'); manifest=read('reports/run_manifest.json')
    experiments=read('reports/experiments/experiments.json'); curve=read('reports/experiments/learning_curve.json'); latency=read('reports/latency.json')
    train=pd.read_csv(ROOT/'data/processed/train.csv'); val=pd.read_csv(ROOT/'data/processed/validation.csv'); test=pd.read_csv(ROOT/'data/processed/test.csv')
    figures=ROOT/'reports/figures'
    plt.rcParams.update({'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False,'axes.titlesize':15,'font.size':11})
    colors=['#8090A5','#22A89E','#304C89']
    fig,ax=plt.subplots(figsize=(8,4.6))
    distributions=pd.DataFrame({name:frame.label.value_counts().reindex(LABEL_ORDER) for name,frame in [('Train',train),('Validation',val),('Test',test)]})
    distributions.plot.bar(ax=ax,color=colors,rot=0)
    ax.set(title='Class distribution by split',ylabel='Resume–JD pairs',xlabel=''); fig.tight_layout(); fig.savefig(figures/'class_distribution.png',dpi=160); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4.1))
    for ax,col,title in zip(axes,['resume_text','jd_text'],['Resume length','JD length']):
        values=train[col].map(clean_text).str.split().str.len()
        ax.hist(values,bins=35,color='#304C89',edgecolor='white'); ax.set(title=title,xlabel='Words',ylabel='Training pairs')
    fig.tight_layout(); fig.savefig(figures/'text_lengths.png',dpi=160); plt.close(fig)
    for name in ['baseline','advanced']:
        plot_confusion(np.array(metrics[name]['confusion_matrix']),f'{name.title()} model: held-out test',figures/f'{name}_confusion_matrix.png')
    plot_confusion(np.array(metrics['advanced']['confusion_matrix']),'Advanced model: held-out test',figures/'confusion_matrix.png')
    fig,ax=plt.subplots(figsize=(8,4.4)); names=list(metrics); x=np.arange(len(names)); width=.35
    ax.bar(x-width/2,[metrics[n]['accuracy'] for n in names],width,label='Accuracy',color='#304C89')
    ax.bar(x+width/2,[metrics[n]['macro_f1'] for n in names],width,label='Macro F1',color='#22A89E')
    ax.set(xticks=x,xticklabels=[n.title() for n in names],ylim=(0,1),title='Same held-out test set',ylabel='Score'); ax.legend()
    fig.tight_layout(); fig.savefig(figures/'model_comparison.png',dpi=160); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.4))
    for key,label,color in [('train_macro_f1','Training','#304C89'),('validation_macro_f1','Validation','#22A89E')]:
        ax.plot([c['rows'] for c in curve],[c[key] for c in curve],marker='o',label=label,color=color)
    ax.set(title='Selected classifier learning curve',xlabel='Training pairs',ylabel='Macro F1',ylim=(0,1)); ax.legend(); fig.tight_layout(); fig.savefig(figures/'learning_curve.png',dpi=160); plt.close(fig)
    rows='\n'.join(f"| {name.title()} | {m['accuracy']:.4f} | {m['macro_f1']:.4f} | {m['weighted_f1']:.4f} | {m['ranking']['ndcg_at_3']:.4f} | {m['ranking']['precision_at_3_strong_fit']:.4f} |" for name,m in metrics.items())
    classrows='\n'.join(f"| {label} | {metrics['advanced']['class_report'][label]['precision']:.4f} | {metrics['advanced']['class_report'][label]['recall']:.4f} | {metrics['advanced']['class_report'][label]['f1-score']:.4f} | {int(metrics['advanced']['class_report'][label]['support'])} |" for label in LABEL_ORDER)
    predictions=pd.read_csv(ROOT/'reports/advanced_test_predictions.csv')
    selected=predictions.groupby('actual',sort=False).head(2)
    errors=predictions[predictions.actual!=predictions.predicted].groupby('actual',sort=False).head(2)
    examples='\n'.join(f"| {r.candidate_name} | {r.actual} | {r.predicted} | {r.score:.2f} |" for r in selected.itertuples())
    errorrows='\n'.join(f"| {r.candidate_name} | {r.actual} | {r.predicted} | {r.score:.2f} |" for r in errors.itertuples())
    delta=metrics['advanced']['macro_f1']-metrics['baseline']['macro_f1']
    model=ResumeJDMatcher.load(ROOT/'models/resume_matcher.joblib')
    demo=read('data/demo_cases.json')
    demo_results=model.rank_candidates(demo['jd_text'],demo['candidates'])
    failure=model.predict_one(demo['failure_case']['resume_text'],demo['jd_text'],'Keyword-only example')
    write_json(ROOT/'reports/demo_results.json',{'ranked_candidates':demo_results,'failure_case':failure})
    fairness=[]
    headers=['Gender: Female\nAge: 26\nMarital status: Single\n','Gender: Male\nAge: 52\nMarital status: Married\n',
             'Religion: Example A\nCaste: Example A\n','Religion: Example B\nCaste: Example B\n']
    for candidate in demo['candidates']:
        outputs=[model.predict_one(header+candidate['resume_text'],demo['jd_text'],candidate['candidate_name']) for header in headers]
        invariant=all(o['probabilities']==outputs[0]['probabilities'] and o['match_score']==outputs[0]['match_score'] for o in outputs)
        fairness.append({'example':candidate['candidate_name'],'variants':len(headers),'invariant':invariant})
    checks=[]
    for resume,jd,matched,missing in [('Python SQL','Python SQL Docker',['python','sql'],['docker']),('Java Spring Boot','Java SQL',['java'],['sql']),('Excel Power BI','Excel Power BI Tableau',['excel','power bi'],['tableau'])]:
        gap=analyze_skill_gap(resume,jd)
        checks.append({'resume':resume,'jd':jd,'expected_matched':matched,'actual_matched':gap.matched_skills,
                       'expected_missing':missing,'actual_missing':gap.missing_skills,
                       'passed':gap.matched_skills==matched and gap.missing_skills==missing})
    write_json(ROOT/'reports/responsible_use_checks.json',{'demographic_field_checks':fairness,'keyword_explanation_checks':checks,
        'limits':'Small controlled regression checks only. Not a demographic fairness certification or a human annotation study. Free-form proxies and negation remain.'})
    write('reports/model_evaluation.md',f"""
    # Model evaluation and error analysis

    Presenter: Sanket Toche. Run configuration and hashes: `run_manifest.json`.

    ## Evaluation protocol
    Source train/test snapshot: `{manifest['dataset']}` at `{manifest['dataset_revision']}`.
    Clean source training pairs: {manifest['splits']['train_full']}. Development split: {len(train)} training pairs and {len(val)} validation pairs, grouped by JD (seed 42).
    Select the advanced configuration by validation macro F1, refit on clean source train, then evaluate on all {len(test)} original test pairs. Test labels never select a configuration.
    Selected model: **{manifest['model']['selected_model']}**. Validation macro F1: **{manifest['model']['validation_macro_f1']:.4f}**.
    The frozen encoder processes up to eight 150-word chunks per document, normalizes their pooled representation, and produces 384-dimensional vectors. No capstone encoder fine-tuning.

    ## Test comparison
    | Model | Accuracy | Macro F1 | Weighted F1 | NDCG@3 | Precision@3 |
    |---|---:|---:|---:|---:|---:|
    {rows}

    Advanced minus TF-IDF baseline macro F1: **{delta:+.4f}**. These are observed point estimates, not a significance claim.
    Ranking metrics average over {metrics['advanced']['ranking']['evaluated_jd_groups']} eligible JD groups. {metrics['advanced']['ranking']['definition']}
    Evaluation and deployment share a composite ranking score: 30% expected fit, 45% clipped embedding cosine (lexical cosine for the baseline) and 25% keyword skill overlap. This fixed engineering ranking rule is not fitted to test labels. It is not a calibrated hiring probability.
    Classifier categories remain separate diagnostics and can disagree with rank. This separation is explicit in the UI; categories are not manually relabeled to match the score.

    ![Model comparison](figures/model_comparison.png)

    ## Advanced model by class
    | Actual class | Precision | Recall | F1 | Support |
    |---|---:|---:|---:|---:|
    {classrows}

    ![Confusion matrix](figures/confusion_matrix.png)

    ## Representative sample predictions
    | Candidate | Actual | Predicted | Priority |
    |---|---|---|---:|
    {examples}

    ## Actual test errors
    | Candidate | Actual | Predicted | Priority |
    |---|---|---|---:|
    {errorrows}

    Every row is traceable to `advanced_test_predictions.csv` and the original test CSV. These examples are selected after evaluation to illustrate errors, not used for tuning.
    All source records currently have a generic job-title field, so reliable job-family segment metrics cannot be claimed.

    ## Learning curve and experiments
    ![Learning curve](figures/learning_curve.png)
    Full configurations, validation metrics and elapsed training time are in `experiments/experiments.json` and `.csv`.
    The curve varies classifier training-set size on one fixed JD-grouped validation split. Encoder weights stay frozen, so optimizer/learning-rate/epoch tuning of the transformer is not claimed.
    The full-size learning-curve point has training macro F1 {curve[-1]['train_macro_f1']:.4f} versus validation macro F1 {curve[-1]['validation_macro_f1']:.4f}. This large gap and the non-monotonic validation curve show overfitting/instability. More labeled data, stronger regularization and grouped cross-validation are future work, not completed claims.

    ## Latency
    Loaded-encoder CPU inference, {latency['samples']} pairs, no per-document cache reuse: median **{latency['median_ms']:.1f} ms**, p95 **{latency['p95_ms']:.1f} ms**.
    Includes embedding generation and skill suggestions. Excludes process/model loading, HTTP/network and concurrent load. Hardware: {latency['cpu']}. Python {latency['python']}.

    ## Explanation review and fairness checks
    Three deterministic keyword fixtures check expected matched/missing lists. Twelve controlled demographic-field variants across three demo resumes check prediction invariance after redaction.
    Results: `responsible_use_checks.json`. This is a small regression audit, not a fairness certification. Protected-group outcome metrics cannot be inferred without suitable group labels and consent.
    Embedding suggestions show possible paraphrases as unverified evidence; keyword matches and skill proficiency remain distinct.

    ## Known failure and intended use
    The keyword-only demo produces `{failure['predicted_fit']}` with priority {failure['match_score']:.2f}; it still detects {len(failure['skill_gap']['matched_skills'])} matching keyword skills despite explicit lack of experience.
    The very short new-input examples can receive overconfident or incorrect categories despite different relevance. The composite ranking adds semantic/skill evidence, but does not repair category validity. The app displays short-input and prediction/similarity-conflict warnings.
    This is a reproducible explanation failure. Negation, keyword stuffing, experience depth and false negatives require manual review.
    Most test resumes ({audit['splits']['test_resumes_also_in_training']}/{len(test)} pairs) also occur in clean training, although test JDs and exact pairs do not. Unseen-candidate generalization remains unproven.
    Dataset card/provenance and synthetic-data status are uncertain. Boilerplate filtering and explicit-field redaction are limited mitigations.
    Use this capstone for a supervised recruiter demonstration. It is not sufficient evidence for autonomous hiring, rejection or production deployment.
    """)
    write('reports/data_quality.md',f"""
    # Dataset audit and preprocessing

    Public source: https://huggingface.co/datasets/{manifest['dataset']}
    Snapshot: `{manifest['dataset_revision']}`. Original brief omits the hyphen in `resume-jd-match`; this correction is intentional.
    Frozen input CSV hashes appear in `run_manifest.json`. `scripts/run_capstone.py --download` reloads the pinned public snapshot.
    The upstream dataset card is empty. We cannot establish label-annotation quality, licensing terms for redistribution, or whether examples are synthetic from the card. Keep public-data provenance uncertainty visible.

    ## Audit results
    ```json
    {json.dumps(audit,indent=2)}
    ```

    Required-column and strict combined-prompt validation reject malformed source records. Labels map No Fit / Potential Fit / Good Fit to low_fit / potential_fit / strong_fit.
    Preprocessing normalizes whitespace/symbols/case, removes selected boilerplate clauses and explicit demographic/contact fields. Free-form identity, proxy bias, negation and qualification depth remain unresolved.
    Conflicting pairs are quarantined using source TRAIN labels only. Exact normalized training duplicates are removed. Test rows and labels remain unchanged.
    Development train and validation share no normalized JDs. No normalized JD or exact pair overlaps with the test split.
    Resume reuse is substantial; the primary experiment measures transfer to new JDs, not to entirely new people.

    ## EDA observations
    Low-fit is the largest class; accuracy alone can reward majority predictions. Macro F1 weights each class equally and is the configuration-selection metric.
    Pair rows reuse a limited pool of resumes and JDs. Grouped validation therefore better reflects the unseen-JD evaluation than a random row split.
    Text lengths vary and long inputs exceed the encoder's 256-wordpiece context. Up to eight spaced 150-word chunks improve coverage but can still truncate tokens or omit sections.
    Keyword taxonomy emphasizes technology and analytics roles, limiting explanation coverage for other domains.

    ![Class distribution](figures/class_distribution.png)
    ![Text lengths](figures/text_lengths.png)
    """)
    write('reports/ethics_and_fairness.md',"""
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
    """)
    notebook('01_data_access_preprocessing.ipynb',[
        ('md','# 01 · Dataset access and preprocessing\nPublic source, strict parsing, label normalization, audits and grouped splits. Execute with the HeroVired (.venv) kernel.'),('code',BOOT),
        ('code',"from resume_matcher.data import DATASET_ID, DATASET_REVISION, load_huggingface_dataset\nprint(DATASET_ID, DATASET_REVISION)\n# Optional network refresh: raw = load_huggingface_dataset(split='train')\nraw=pd.read_csv(ROOT/'data/raw/original_train.csv')\nraw.shape, raw.columns.tolist()"),
        ('code',"from resume_matcher.research import audit_frame, clean_training_data, grouped_split\nfrom resume_matcher.preprocessing import clean_text\nprint(json.dumps(audit_frame(raw),indent=2))\nclean, quarantine=clean_training_data(raw)\ntrain, validation=grouped_split(clean)\nprint({'clean_pairs':len(clean),'quarantined_rows':len(quarantine),'train':len(train),'validation':len(validation)})"),
        ('code',"assert not set(train.jd_text.map(clean_text)) & set(validation.jd_text.map(clean_text))\ntest=pd.read_csv(ROOT/'data/raw/original_test.csv')\nassert not set(clean.jd_text.map(clean_text)) & set(test.jd_text.map(clean_text))\nprint('JD groups are disjoint. Test rows:',len(test))\nprint('Normalized sample:',clean_text(raw.iloc[0].resume_text)[:250])"),
        ('md','Conflicting TRAIN labels never consult test labels. The upstream card is empty, so synthetic-data status and annotation provenance remain uncertain. Reused resumes limit claims about unseen candidates.')])
    notebook('02_eda.ipynb',[
        ('md','# 02 · Exploratory analysis and data quality\nClass imbalance, document length, repeated entities, conflicting labels and split overlap.'),('code',BOOT),
        ('code',"from IPython.display import display, Image\naudit=read_json('reports/data_audit.json')\ndisplay(pd.DataFrame({name:{'rows':a['rows'],'unique_jds':a['unique_jds'],'unique_resumes':a['unique_resumes'],'duplicate_pairs':a['duplicate_pairs'],'conflicting_pairs':a['conflicting_pairs']} for name,a in audit.items() if name!='splits'}).T)\nprint(audit['splits'])"),
        ('code',"train=pd.read_csv(ROOT/'data/processed/train.csv')\nfrom resume_matcher.preprocessing import clean_text\ndisplay(train.label.value_counts().to_frame('pairs'))\ndisplay(pd.DataFrame({c:train[c].map(clean_text).str.split().str.len().describe() for c in ['resume_text','jd_text']}))"),
        ('code',"display(Image(filename=str(ROOT/'reports/figures/class_distribution.png')))\ndisplay(Image(filename=str(ROOT/'reports/figures/text_lengths.png')))"),
        ('md','Low-fit dominates the labels. Use macro F1 alongside accuracy. Pair reuse makes random row splits optimistic for JD transfer. Long documents motivate chunked embeddings. Six contradictory pair groups require quarantine. The skill taxonomy is narrower than the dataset’s role coverage.')])
    notebook('03_baseline_model.ipynb',[
        ('md','# 03 · TF-IDF and Random Forest baseline\nTrain on development training rows, evaluate on JD-grouped validation. The final refitted model has a separate untouched-test report.'),('code',BOOT),
        ('code',"from resume_matcher.model import ResumeJDMatcher\nfrom sklearn.metrics import classification_report\ntrain=pd.read_csv(ROOT/'data/processed/train.csv')\nvalidation=pd.read_csv(ROOT/'data/processed/validation.csv')\nbaseline=ResumeJDMatcher().fit(train)\nx=baseline._feature_matrix(validation.resume_text,validation.jd_text)\npred=baseline.label_encoder.inverse_transform(baseline.classifier.predict(x))\nprint(classification_report(validation.label,pred,zero_division=0))\nprint(baseline.classifier.get_params())"),
        ('code',"metrics=read_json('reports/metrics.json')\nprint('Final source-train refit, untouched test:',{k:metrics['baseline'][k] for k in ['accuracy','macro_f1','weighted_f1']})\nfrom IPython.display import display, Image\ndisplay(Image(filename=str(ROOT/'reports/figures/baseline_confusion_matrix.png')))"),
        ('md','TF-IDF captures lexical overlap. Eight numeric features summarize cosine, skill/token overlap and relative lengths. Class balancing cannot remove label noise or prove fairness. The majority baseline is included to avoid interpreting accuracy in isolation.')])
    notebook('04_advanced_model.ipynb',[
        ('md','# 04 · Sentence-BERT advanced model and experiment comparison\nFrozen pretrained encoder with a fitted paired-document classifier. No claim of transformer fine-tuning.'),('code',BOOT),
        ('code',"encoder=read_json('models/encoder_manifest.json')\nprint(encoder)\nexperiments=pd.read_csv(ROOT/'reports/experiments/experiments.csv')\ndisplay(experiments.sort_values('validation_macro_f1',ascending=False))"),
        ('code',"from resume_matcher.model import ResumeJDMatcher\nfrom resume_matcher.advanced import EmbeddingResumeJDMatcher\nfrom sklearn.base import clone\nfrom sklearn.metrics import classification_report\nimport joblib\nfinal=ResumeJDMatcher.load(ROOT/'models/resume_matcher.joblib')\ntrain=pd.read_csv(ROOT/'data/processed/train.csv')\nvalidation=pd.read_csv(ROOT/'data/processed/validation.csv')\nadvanced=EmbeddingResumeJDMatcher(clone(final.classifier))\nadvanced.embedding_cache=joblib.load(ROOT/'models/document_embeddings.joblib')['embeddings']\nadvanced.fit(train)\nx=advanced._feature_matrix(validation.resume_text,validation.jd_text)\npred=advanced.label_encoder.inverse_transform(advanced.classifier.predict(x))\nprint(classification_report(validation.label,pred,zero_division=0))\nprint('Feature dimensions:',x.shape[1])"),
        ('code',"metrics=read_json('reports/metrics.json')\ndisplay(pd.DataFrame({name:{'accuracy':m['accuracy'],'macro_F1':m['macro_f1'],'NDCG@3':m['ranking']['ndcg_at_3'],'Precision@3':m['ranking']['precision_at_3_strong_fit']} for name,m in metrics.items()}).T)\nfrom IPython.display import Image\ndisplay(Image(filename=str(ROOT/'reports/figures/model_comparison.png')))\ndisplay(Image(filename=str(ROOT/'reports/figures/learning_curve.png')))"),
        ('code',"demo=read_json('data/demo_cases.json')\nresults=final.rank_candidates(demo['jd_text'],demo['candidates'])\ndisplay(pd.DataFrame([{k:r[k] for k in ['candidate_name','predicted_fit','match_score']} for r in results]))\nfailure=final.predict_one(demo['failure_case']['resume_text'],demo['jd_text'])\nprint('Known explanation failure:',failure['predicted_fit'],failure['skill_gap']['matched_skills'])\nprint(read_json('reports/responsible_use_checks.json'))"),
        ('md','Configuration selection uses validation macro F1. Test metrics report the final refit and do not select a configuration. Ranking relevance is 0/1/2 and Precision@3 counts strong fit. Score probabilities remain uncalibrated. Shared resumes, uncertain label provenance and limited explanation checks constrain business use.')])
    write('notebooks/README.md','''
    # Executable capstone notebooks
    Run notebooks 01–04 using the HeroVired (.venv) kernel. They contain executed outputs after `scripts/execute_notebooks.py`.
    Data access/preprocessing, EDA, baseline training and advanced classifier training are all included.
    Cached document embeddings and local encoder weights keep notebook reruns offline. Full experiment reproduction: `python scripts/run_capstone.py`.
    ''')
    write('presentation/final_presentation.md',f'''
    # Final eight-slide presentation · Sanket Toche

    Editable deck: `output/HeroVired_Capstone_Sanket_Toche.pptx`. Speaker notes accompany every slide.

    1. **Title and problem context.** Resume Screening and Job Description Matching. Recruiter decision support and explainable candidate comparison.
    2. **Dataset source and access.** Pinned Hugging Face snapshot. 8,000 original pairs; {manifest['splits']['train_full']} clean source training pairs; {len(train)} development training, {len(val)} validation and {len(test)} original test pairs. Illustrative schema example.
    3. **EDA and preprocessing.** Class distribution, conflicting labels, normalization, redaction, document lengths and chunking.
    4. **Baseline and advanced architecture.** TF-IDF + Random Forest versus frozen Sentence-BERT paired embeddings + selected classifier. The encoder is not fine-tuned on capstone labels.
    5. **Experiments and hyperparameters.** Validation comparison, regularization/leaf-size settings, selection by macro F1, learning curve and frozen encoder settings.
    6. **Evaluation and error analysis.** Advanced accuracy {metrics['advanced']['accuracy']:.4f}, macro F1 {metrics['advanced']['macro_f1']:.4f}, NDCG@3 {metrics['advanced']['ranking']['ndcg_at_3']:.4f}; native confusion matrix and latency evidence.
    7. **Deployment demo.** Saved local model, three actual predictions, ranking CSV and keyword-only explanation failure.
    8. **Risks and future work.** Human review, controlled redaction checks, uncertain source labels, shared resumes, unseen-candidate validation and recruiter feedback.

    Measured test comparison:

    | Model | Accuracy | Macro F1 | Weighted F1 | NDCG@3 | Precision@3 |
    |---|---:|---:|---:|---:|---:|
    {rows}
    ''')
    print('Reports, figures, responsible-use checks, demo results and four notebooks generated.',flush=True)

if __name__=='__main__': main()
