"""Offline inference API. Launch with the project's .venv Python."""
from __future__ import annotations
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, StringConstraints
from resume_matcher.model import ResumeJDMatcher
from resume_matcher.advanced import EmbeddingResumeJDMatcher, load_encoder

ROOT=Path(__file__).resolve().parent
MODEL_PATH=ROOT/'models/resume_matcher.joblib'
model=None

@asynccontextmanager
async def lifespan(app):
    global model
    model=None
    if MODEL_PATH.exists():
        model=ResumeJDMatcher.load(MODEL_PATH)
        if isinstance(model,EmbeddingResumeJDMatcher): load_encoder()
    yield
    model=None

app=FastAPI(title='Resume-JD Matcher API',version='2.0.0',lifespan=lifespan,
            description='Recruiter decision support. Scores blend fit, similarity and skills; classifier categories are separate diagnostics. Scores are not calibrated hiring probabilities.')
InputText=Annotated[str,StringConstraints(strip_whitespace=True,min_length=20,max_length=50000)]

class Candidate(BaseModel):
    candidate_name:str=Field(default='Candidate',min_length=1,max_length=100)
    resume_text:InputText

class MatchRequest(BaseModel):
    jd_text:InputText
    candidates:list[Candidate]=Field(min_length=1,max_length=50)

@app.get('/health')
def health():
    if model is None: raise HTTPException(status_code=503,detail='Model artifact is not loaded')
    return {'status':'ok','model_loaded':True,'model':getattr(model,'metadata',{}).get('selected_model','TF-IDF baseline')}

@app.post('/match')
def match(request:MatchRequest):
    if model is None: raise HTTPException(status_code=503,detail='Model is unavailable')
    try:
        ranked=model.rank_candidates(request.jd_text,[c.model_dump() for c in request.candidates])
    except ValueError as exc:
        raise HTTPException(status_code=422,detail=str(exc)) from exc
    return {'ranked_candidates':ranked,'responsible_use':'Human review required. Never reject candidates solely from this output.'}
