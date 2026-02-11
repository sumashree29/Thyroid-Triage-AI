@echo off
TITLE Thyroid Triage AI
echo ====================================================
echo      THYROID TRIAGE AI - SETUP & RUN
echo ====================================================
echo.
echo [1/2] Checking dependencies...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo Error installing dependencies.
    pause
    exit /b %errorlevel%
)

echo.
echo [2/2] Starting server...
echo.
echo Open your browser to: http://localhost:8000
echo.
python api.py
pause
