@echo off
title SafeGuard AI v2 — Industrial Safety Monitor
color 03
cls

:: ── Python 3.12 (CUDA + all packages) ────────────────────────────────────────
set PYTHON_DIR=C:\Users\kbhas\AppData\Local\Programs\Python\Python312
set PIP=%PYTHON_DIR%\Scripts\pip.exe
set PY=%PYTHON_DIR%\python.exe
set PATH=%PYTHON_DIR%;%PYTHON_DIR%\Scripts;%PATH%

echo.
echo  ╔══════════════════════════════════════════════════════════════════╗
echo  ║        SafeGuard AI  —  Industrial Safety Monitor  v2.0         ║
echo  ║      YOLOv11n  ·  PPE Detection  ·  Tool Abandonment FSM        ║
echo  ╚══════════════════════════════════════════════════════════════════╝
echo.
echo  [1/3] Checking Python environment...
echo        Python: %PYTHON_DIR%
echo.

:: Verify Python exists
if not exist "%PY%" (
    echo  [ERROR] Python not found at: %PY%
    echo          Edit PYTHON_DIR in this bat file to point to your Python 3.12.
    pause >nul
    exit /b 1
)

echo  [2/3] Installing / updating required packages...
echo        (First run: ~2 min  ·  Subsequent runs: instant)
echo.

"%PIP%" install --quiet --upgrade pip
"%PIP%" install --quiet --upgrade ^
    ultralytics streamlit torch torchvision ^
    opencv-python numpy pandas plotly ^
    fpdf2 psycopg2-binary Pillow requests ^
    streamlit-option-menu

echo.
echo  [3/3] Launching SafeGuard AI dashboard...
echo.
echo  ┌─────────────────────────────────────────┐
echo  │  URL :  http://localhost:8501            │
echo  │  Stop:  Close this window               │
echo  └─────────────────────────────────────────┘
echo.

:: Open browser after 5 seconds
timeout /t 5 /nobreak >nul
start "" "http://localhost:8501"

:: ── Launch the new cyberpunk dashboard ────────────────────────────────────────
cd /d "E:\4TH YEAR PROJECT\WEB DEPLOYMENT"
"%PY%" -m streamlit run streamlit_app.py ^
    --server.port 8501 ^
    --server.headless true ^
    --browser.gatherUsageStats false ^
    --theme.base dark

pause >nul
