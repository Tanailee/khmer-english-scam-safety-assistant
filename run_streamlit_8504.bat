@echo off
cd /d "%~dp0"
echo Starting Scam or Not Scam Detector on http://localhost:8504
python -m streamlit run app\streamlit_app.py --server.port 8504 --server.headless true --server.fileWatcherType none --browser.gatherUsageStats false
pause
