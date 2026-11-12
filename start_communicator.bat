@echo off
cd /d "%~dp0"
echo ========================================================
echo   Starting ASL Communicator V10 (Hardware Edition)
echo ========================================================
.\.venv\Scripts\python.exe live_pipeline.py %*
pause
