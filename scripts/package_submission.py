"""Create a portable source/artifact ZIP without virtualenv or private build files."""
from pathlib import Path
from importlib.metadata import version, PackageNotFoundError
import json, platform, zipfile
from resume_matcher.research import ROOT, sha256, write_json

def main():
    packages=['numpy','pandas','scipy','scikit-learn','joblib','torch','transformers','tokenizers','sentence-transformers','huggingface-hub','datasets','streamlit','fastapi','uvicorn','pydantic','matplotlib','nbformat','nbclient','ipykernel','pytest','httpx']
    environment={'python':platform.python_version(),'platform':platform.platform(),'cpu':platform.processor(),
        'packages':{p:version(p) for p in packages},'scope':'Measured direct dependency versions. No claim of a fully isolated or transitive hash-locked environment.'}
    write_json(ROOT/'reports/environment.json',environment)
    required=['presentation/output/HeroVired_Capstone_Sanket_Toche.pptx','reports/metrics.json','reports/run_manifest.json',
              'reports/requirements_alignment.md','models/resume_matcher.joblib','models/encoder/modules.json']
    for path in required:
        if not (ROOT/path).exists(): raise FileNotFoundError(path)
    files=[]
    for folder in ['src/resume_matcher','scripts','tests','notebooks','reports','presentation','data','models/encoder']:
        for path in (ROOT/folder).rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts and '.ipynb_checkpoints' not in path.parts and path.suffix!='.pyc':
                if path.name=='original_model_evaluation.md': continue
                if path==ROOT/'reports/submission_manifest.json': continue
                files.append(path)
    for path in ['README.md','requirements.txt','requirements-lock.txt','pyproject.toml','.gitignore','.gitattributes','.dockerignore','Dockerfile',
                 'app_streamlit.py','api_fastapi.py','launch_dashboard.cmd','launch_api.cmd','models/resume_matcher.joblib',
                 'models/baseline.joblib','models/encoder_manifest.json','models/document_embeddings.joblib']:
        files.append(ROOT/path)
    files=sorted(set(files))
    write_json(ROOT/'reports/submission_manifest.json',{'files':{str(p.relative_to(ROOT)).replace('\\','/'):sha256(p) for p in files}})
    files.append(ROOT/'reports/submission_manifest.json')
    destination=ROOT/'submission'; destination.mkdir(exist_ok=True)
    zip_path=destination/'HeroVired_Capstone_Sanket_Toche.zip'
    with zipfile.ZipFile(zip_path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=5) as archive:
        for path in files:
            archive.write(path,'HeroVired_Capstone/'+str(path.relative_to(ROOT)).replace('\\','/'))
    with zipfile.ZipFile(zip_path) as archive:
        assert archive.testzip() is None
    print('Submission ZIP:',zip_path,'bytes:',zip_path.stat().st_size,'files:',len(files),flush=True)

if __name__=='__main__':main()
