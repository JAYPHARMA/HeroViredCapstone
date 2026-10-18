"""Reproducible audits, grouped splits and metrics shared by scripts/notebooks."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, ndcg_score
from sklearn.model_selection import GroupShuffleSplit
from .model import LABEL_ORDER
from .preprocessing import clean_text

ROOT = Path(__file__).resolve().parents[2]

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def frame_keys(frame):
    return pd.Series([hashlib.sha256((clean_text(r)+'\x1f'+clean_text(j)).encode()).hexdigest()
                      for r,j in zip(frame.resume_text, frame.jd_text)], index=frame.index)

def audit_frame(frame):
    keys = frame_keys(frame)
    conflicts = frame.assign(pair_key=keys).groupby('pair_key').label.nunique()
    return {'rows':len(frame), 'nulls':frame.isna().sum().astype(int).to_dict(),
            'empty_text':{c:int(frame[c].fillna('').map(clean_text).eq('').sum()) for c in ['resume_text','jd_text']},
            'labels':frame.label.value_counts().astype(int).to_dict(),
            'duplicate_pairs':int(keys.duplicated().sum()), 'conflicting_pairs':int((conflicts>1).sum()),
            'unique_jds':int(frame.jd_text.map(clean_text).nunique()), 'unique_resumes':int(frame.resume_text.map(clean_text).nunique())}

def clean_training_data(frame):
    """Quarantine ambiguous TRAIN pairs only; never consult test labels."""
    data = frame.dropna(subset=['resume_text','jd_text','label']).copy()
    data = data[data.label.isin(LABEL_ORDER)]
    data = data[data.resume_text.map(clean_text).ne('') & data.jd_text.map(clean_text).ne('')]
    data['pair_key'] = frame_keys(data)
    conflicts = data.groupby('pair_key').label.nunique()
    bad = set(conflicts[conflicts>1].index)
    quarantined = data[data.pair_key.isin(bad)].copy()
    clean = data[~data.pair_key.isin(bad)].drop_duplicates('pair_key').drop(columns='pair_key').reset_index(drop=True)
    return clean, quarantined

def grouped_split(frame):
    groups = frame.jd_text.map(clean_text)
    train_idx, validation_idx = next(GroupShuffleSplit(n_splits=1, test_size=.20, random_state=42).split(frame, groups=groups))
    train, validation = frame.iloc[train_idx].copy(), frame.iloc[validation_idx].copy()
    assert not set(train.jd_text.map(clean_text)) & set(validation.jd_text.map(clean_text))
    assert set(train.label) == set(validation.label) == set(LABEL_ORDER)
    return train.reset_index(drop=True), validation.reset_index(drop=True)

def priority_scores(probabilities, semantic_similarity=None, skill_overlap=None):
    """Classes are explicitly ordered low, potential, strong."""
    probabilities=np.asarray(probabilities)
    rounded = np.round(probabilities, 4)
    expected = (rounded[:,1]+2*rounded[:,2])/2
    semantic=np.zeros(len(probabilities)) if semantic_similarity is None else np.round(semantic_similarity,4)
    skills=np.zeros(len(probabilities)) if skill_overlap is None else skill_overlap
    score=.30*expected+.45*np.clip(semantic,0,1)+.25*skills
    return np.array([round(float(s)*100,2) for s in score])

def ranking_metrics(frame, scores, k=3):
    data=frame.copy(); data['priority']=scores
    ndcgs, precisions=[],[]
    skipped=0
    for _, group in data.groupby(data.jd_text.map(clean_text)):
        relevance = group.label.map({'low_fit':0,'potential_fit':1,'strong_fit':2}).to_numpy()
        s = group.priority.to_numpy(); cutoff=min(k,len(group))
        if len(group)<2 or relevance.max()==0:
            skipped+=1; continue
        ndcgs.append(float(ndcg_score([relevance],[s],k=cutoff)))
        # Expected precision at tied cutoff, independent of source row ordering.
        threshold=np.sort(s)[-cutoff]
        above=s>threshold; tied=s==threshold
        relevant=(relevance==2)
        precisions.append(float((relevant[above].sum()+(cutoff-above.sum())*relevant[tied].mean())/cutoff))
    return {'ndcg_at_3':float(np.mean(ndcgs)), 'precision_at_3_strong_fit':float(np.mean(precisions)),
            'evaluated_jd_groups':len(ndcgs),'skipped_groups':skipped,
            'definition':'Macro over JD groups with >=2 candidates and at least one non-low fit; NDCG relevance 0/1/2; Precision@3 relevant=strong_fit; tie-aware.'}

def evaluate(frame, predictions, scores):
    return {'accuracy':float(accuracy_score(frame.label,predictions)),
            'macro_f1':float(f1_score(frame.label,predictions,average='macro',zero_division=0)),
            'weighted_f1':float(f1_score(frame.label,predictions,average='weighted',zero_division=0)),
            'class_report':classification_report(frame.label,predictions,labels=LABEL_ORDER,output_dict=True,zero_division=0),
            'confusion_matrix':confusion_matrix(frame.label,predictions,labels=LABEL_ORDER).tolist(),
            'ranking':ranking_metrics(frame,scores)}

def write_json(path, value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,indent=2,default=str),encoding='utf-8')
