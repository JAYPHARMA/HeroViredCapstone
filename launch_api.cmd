@echo off
cd /d "%~dp0"
".venv\Scripts\python.exe" -m uvicorn api_fastapi:app --host 127.0.0.1 --port 8000
pause
