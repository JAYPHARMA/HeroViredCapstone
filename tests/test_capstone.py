import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from resume_matcher.model import ResumeJDMatcher
from resume_matcher.preprocessing import clean_text
from resume_matcher.research import ROOT, grouped_split, clean_training_data, priority_scores, ranking_metrics
from resume_matcher.skills import analyze_skill_gap
from resume_matcher.data import _split_hf_prompt

@pytest.fixture(scope='module')
def model(): return ResumeJDMatcher.load(ROOT/'models/resume_matcher.joblib')

def test_dataset_parser_rejects_malformed_prompt():
    with pytest.raises(ValueError): _split_hf_prompt('missing resume marker')
    assert _split_hf_prompt('For the given job description <<Python JD>> and the resume: <<SQL resume>>') == ('Python JD','SQL resume')

def test_conflicts_quarantined_without_consulting_test():
    df=pd.DataFrame({'resume_text':['Python','Python','SQL','SQL'], 'jd_text':['JD','JD','Role','Role'],
                     'label':['low_fit','strong_fit','potential_fit','potential_fit']})
    clean,bad=clean_training_data(df)
    assert len(bad)==2 and len(clean)==1 and clean.iloc[0].label=='potential_fit'

def test_jd_group_integrity():
    train=pd.read_csv(ROOT/'data/processed/train.csv')
    val=pd.read_csv(ROOT/'data/processed/validation.csv')
    test=pd.read_csv(ROOT/'data/processed/test.csv')
    assert not set(train.jd_text.map(clean_text)) & set(val.jd_text.map(clean_text))
    assert not set(pd.concat([train.jd_text,val.jd_text]).map(clean_text)) & set(test.jd_text.map(clean_text))

def test_priority_blend_rewards_similarity_and_skill_evidence():
    probs=np.array([[.01,.98,.01],[.01,.98,.01],[.98,.01,.01]])
    scores=priority_scores(probs,np.array([.9,.1,.6]),np.array([1.,0.,.5]))
    assert all(0<=s<=100 for s in scores)
    assert scores[0]>scores[2]>scores[1]

def test_ranking_metric_ties_do_not_depend_on_source_order():
    df=pd.DataFrame({'jd_text':['same job']*4,'label':['low_fit','potential_fit','strong_fit','strong_fit']})
    one=ranking_metrics(df,np.zeros(4)); two=ranking_metrics(df.iloc[::-1],np.zeros(4))
    assert one==two and one['precision_at_3_strong_fit']==.5

def test_demographic_contact_fields_are_removed():
    base='Python engineer builds SQL services.'
    variants=['Name: Person A\nGender: Female\nAge: 29\nReligion: Example\nMarital status: Married\n',
              'Name: Person B\nGender: Male\nAge: 52\nCaste: Example\nMarital status: Single\n']
    assert clean_text(variants[0]+base)==clean_text(variants[1]+base)==clean_text(base)
    assert '@' not in clean_text('Contact example@test.com, +91 98765 43210\n'+base)

def test_boilerplate_does_not_remove_whole_flattened_jd():
    assert 'python' in clean_text('Python developer required. We are an equal opportunity employer. Apply now.')

def test_keyword_explanations_and_negation_failure():
    gap=analyze_skill_gap('Python SQL','Python SQL Docker')
    assert gap.matched_skills==['python','sql'] and gap.missing_skills==['docker']
    assert 'python' in analyze_skill_gap('No Python experience','Python engineer').matched_skills

def test_model_has_no_serialized_resume_cache(model):
    assert model.embedding_cache=={}
    assert model.metadata['encoder_frozen'] is True

def test_demographic_invariance_in_predictions(model):
    demo=json.loads((ROOT/'data/demo_cases.json').read_text())
    text=demo['candidates'][0]['resume_text']
    a=model.predict_one('Gender: Female\nAge: 26\n'+text,demo['jd_text'],'Person A')
    b=model.predict_one('Gender: Male\nAge: 51\n'+text,demo['jd_text'],'Person B')
    assert a['probabilities']==b['probabilities'] and a['match_score']==b['match_score']

def test_api_matches_new_inputs_and_rejects_empty():
    from api_fastapi import app
    demo=json.loads((ROOT/'data/demo_cases.json').read_text())
    with TestClient(app) as client:
        assert client.get('/health').status_code==200
        response=client.post('/match',json={'jd_text':demo['jd_text'],'candidates':demo['candidates']})
        assert response.status_code==200
        rows=response.json()['ranked_candidates']
        assert len(rows)==3 and all(0<=r['match_score']<=100 for r in rows)
        assert [r['match_score'] for r in rows]==sorted([r['match_score'] for r in rows],reverse=True)
        for payload in [{'jd_text':' ','candidates':[{'resume_text':' '}]},
                        {'jd_text':demo['jd_text'],'candidates':[]}]:
            assert client.post('/match',json=payload).status_code==422

def test_save_reload_roundtrip_preserves_predictions(model,tmp_path):
    demo=json.loads((ROOT/'data/demo_cases.json').read_text())
    args=(demo['candidates'][0]['resume_text'],demo['jd_text'])
    before=model.predict_one(*args)
    path=tmp_path/'roundtrip.joblib'; model.save(path)
    after=ResumeJDMatcher.load(path).predict_one(*args)
    assert before==after

def test_offline_priority_matches_deployed_scoring(model):
    demo=json.loads((ROOT/'data/demo_cases.json').read_text())
    for candidate in demo['candidates']:
        x=model._feature_matrix([candidate['resume_text']],[demo['jd_text']])
        p=model.classifier.predict_proba(x)
        item=model.predict_one(candidate['resume_text'],demo['jd_text'])
        assert priority_scores(p,x[:,0],x[:,2])[0]==item['match_score']

def test_short_demo_ranking_uses_semantic_tiebreaker(model):
    demo=json.loads((ROOT/'data/demo_cases.json').read_text())
    results=model.rank_candidates(demo['jd_text'],demo['candidates'])
    assert results[0]['candidate_name']=='Relevant ML experience'
    assert results[-1]['candidate_name']=='Unrelated design experience'
    assert all(r['input_warnings'] for r in results)

def test_streamlit_dashboard_and_invalid_form():
    from streamlit.testing.v1 import AppTest
    at=AppTest.from_file(str(ROOT/'app_streamlit.py')).run(timeout=90)
    assert not at.exception
    assert len(at.tabs)==3 and len(at.dataframe[0].value)==3
    at.text_area[0].set_value(' ')
    next(b for b in at.button if b.label=='Rank candidates').click().run(timeout=90)
    assert not at.exception and len(at.error)>0
    assert len(at.dataframe[0].value)==0
