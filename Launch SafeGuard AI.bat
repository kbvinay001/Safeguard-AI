@echo off
title SafeGuard AI — Safety Monitoring System
color 0B
cls

:: ── Python 3.12 (has CUDA + all our packages) ────────────────────────────────
set PYTHON_DIR=C:\Users\kbhas\AppData\Local\Programs\Python\Python312
set PIP=%PYTHON_DIR%\Scripts\pip.exe
set PY=%PYTHON_DIR%\python.exe
set PATH=%PYTHON_DIR%;%PYTHON_DIR%\Scripts;%PATH%

echo.
echo  ══════════════════════════════════════════════════════════════════
echo     SafeGuard AI  —  Safety Monitoring System
echo  ══════════════════════════════════════════════════════════════════
echo.
echo  [1/2] Checking and installing required packages...
echo        (First run may take 1-2 minutes — won't repeat next time)
echo.

:: Auto-install all required packages silently
"%PIP%" install --quiet --upgrade pip
"%PIP%" install --quiet ultralytics streamlit torch torchvision ^
    opencv-python numpy pandas plotly reportlab Pillow requests ^
    streamlit-option-menu

echo.
echo  [2/2] All packages ready. Starting SafeGuard AI...
echo.

:: Open browser after 4s
timeout /t 4 /nobreak >nul
start "" "http://localhost:8501"

:: Launch unified Safety Monitoring Website
cd /d "E:\4TH YEAR PROJECT\WEB DEPLOYMENT"
"%PY%" -m streamlit run safety_monitoring_website.py --server.port 8501 --server.headless true

pause >nul
