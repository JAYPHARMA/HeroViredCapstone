"""Recompute final metrics with the exact deployed score precision; no fitting."""
import json
import joblib
import numpy as np
import pandas as pd
from resume_matcher.model import ResumeJDMatcher, LABEL_ORDER
from resume_matcher.research import ROOT, evaluate, priority_scores, write_json

def main():
    test=pd.read_csv(ROOT/'data/processed/test.csv')
    clean=pd.read_csv(ROOT/'data/processed/train_full.csv')
    result={}
    for name in ['majority','baseline','advanced']:
        if name=='majority':
            pred=np.repeat(clean.label.mode()[0],len(test)); scores=np.zeros(len(test))
        else:
            path='baseline.joblib' if name=='baseline' else 'resume_matcher.joblib'
            model=ResumeJDMatcher.load(ROOT/'models'/path)
            if name=='advanced': model.embedding_cache=joblib.load(ROOT/'models/document_embeddings.joblib')['embeddings']
            x=model._feature_matrix(test.resume_text,test.jd_text)
            p=model.classifier.predict_proba(x)
            indices=[list(model.label_encoder.classes_).index(label) for label in LABEL_ORDER]
            p=p[:,indices]; pred=np.array(LABEL_ORDER)[p.argmax(axis=1)]
            scores=priority_scores(p,x[:,0],x[:,2] if name=='advanced' else x[:,1])
        result[name]=evaluate(test,pred,scores)
        pd.DataFrame({'candidate_name':test.candidate_name,'actual':test.label,'predicted':pred,'score':scores}).to_csv(ROOT/f'reports/{name}_test_predictions.csv',index=False)
    write_json(ROOT/'reports/metrics.json',result)
    manifest=json.loads((ROOT/'reports/run_manifest.json').read_text())
    manifest['deployment_score_policy']='Composite ranking: 30% expected fit, 45% clipped cosine similarity, 25% keyword overlap. Classifier label is preserved separately and may disagree with ranking. Probabilities/cosine rounded to 4 decimals and score to 2, matching deployment.'
    write_json(ROOT/'reports/run_manifest.json',manifest)
    print(json.dumps({n:{k:m[k] for k in ['accuracy','macro_f1','ranking']} for n,m in result.items()},indent=2))

if __name__=='__main__':main()
