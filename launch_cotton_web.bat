@echo off
title Cottonbot Web Console
echo ====================================================
echo      Launching Cottonbot Web Interface...
echo ====================================================
echo.

REM Activate venv if you use one — uncomment and edit this line:
REM call venv\Scripts\activate

REM Optional: check if Flask is installed
python -c "import flask" 2>nul
if errorlevel 1 (
    echo Flask not found. Installing...
    pip install flask
)

REM Run the Flask web app
echo Launching wrapper.py...
python wrapper.py

echo.
echo ====================================================
echo Cottonbot has retired to its coral chambers.
echo Press any key to close this window.
pause >nul
