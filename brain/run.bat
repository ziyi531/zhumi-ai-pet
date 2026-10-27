@echo off
cd /d "%~dp0"
pip install -r requirements.txt
echo Open http://localhost:8600
python -m uvicorn server:app --port 8600
