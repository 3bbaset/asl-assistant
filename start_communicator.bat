@echo off
cd /d "%~dp0"
echo Starting ASL Communicator V9 Live System...
.\.venv\Scripts\python.exe live_pipeline.py
pause
