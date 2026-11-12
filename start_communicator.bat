@echo off
cd /d "%~dp0"
echo ========================================================
echo   ASL AI engine (laptop). The device runs on the Pi.
echo ========================================================
.\.venv\Scripts\python.exe ai_engine.py %*
pause
