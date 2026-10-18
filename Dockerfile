FROM python:3.11-slim
WORKDIR /app
COPY requirements-lock.txt pyproject.toml ./
COPY src ./src
RUN python -m pip install --no-cache-dir torch==2.1.1 --index-url https://download.pytorch.org/whl/cpu && python -m pip install --no-cache-dir -r requirements-lock.txt && python -m pip install --no-deps .
COPY models ./models
COPY data/demo_cases.json ./data/demo_cases.json
COPY reports ./reports
COPY app_streamlit.py api_fastapi.py ./
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
EXPOSE 8501
CMD ["python", "-m", "streamlit", "run", "app_streamlit.py", "--server.address=0.0.0.0", "--server.headless=true"]
