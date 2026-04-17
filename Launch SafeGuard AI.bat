@echo off
title SafeGuard AI v2 -- Industrial Safety Monitor (GPU)
color 0A
cls

:: ============================================================
::  SafeGuard AI Launch Script
::  Uses Anaconda Python 3.13 with PyTorch 2.11+cu128 (RTX 4060)
::  NO package installs on startup -- launches instantly
:: ============================================================

:: Set KMP fix to avoid OpenMP DLL conflict (Anaconda + MKL)
set KMP_DUPLICATE_LIB_OK=TRUE
set CUDA_VISIBLE_DEVICES=0

:: ── Point to Anaconda Python (has CUDA torch 2.11+cu128) ─────────────
set ANACONDA_DIR=C:\Users\kbhas\anaconda3
set PY=%ANACONDA_DIR%\python.exe
set STREAMLIT=%ANACONDA_DIR%\Scripts\streamlit.exe
set PATH=%ANACONDA_DIR%;%ANACONDA_DIR%\Scripts;%ANACONDA_DIR%\Library\bin;%PATH%

echo.
echo  +==================================================================+
echo  ^|       SafeGuard AI  --  Industrial Safety Monitor  v2.0         ^|
echo  ^|     YOLOv11n  .  PPE Detection  .  Tool Abandonment FSM         ^|
echo  ^|        GPU: NVIDIA RTX 4060  ^|  CUDA 12.8  ^|  PyTorch 2.11     ^|
echo  +==================================================================+
echo.

:: ── Verify Anaconda Python exists ─────────────────────────────────────
if not exist "%PY%" (
    echo  [ERROR] Anaconda Python not found at: %PY%
    echo          Check that Anaconda is installed at C:\Users\kbhas\anaconda3
    pause >nul
    exit /b 1
)

:: ── Quick GPU sanity check ────────────────────────────────────────────
echo  [CHECK] Verifying GPU / CUDA...
"%PY%" -c "import os; os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'; import torch; cuda=torch.cuda.is_available(); gpu=torch.cuda.get_device_name(0) if cuda else 'NOT AVAILABLE'; print('  PyTorch :', torch.__version__); print('  CUDA    :', cuda); print('  GPU     :', gpu)"
echo.

:: ── Launch Streamlit ──────────────────────────────────────────────────
echo  [START] Launching SafeGuard AI dashboard...
echo.
echo  +--------------------------------------+
echo  ^|  URL  :  http://localhost:8501       ^|
echo  ^|  Stop :  Press Ctrl+C in this window^|
echo  +--------------------------------------+
echo.

:: Open browser after 4 seconds (background)
start "" /b timeout /t 4 /nobreak >nul
start "" "http://localhost:8501"

:: ── Run the dashboard from WEB DEPLOYMENT dir ─────────────────────────
cd /d "E:\4TH YEAR PROJECT\WEB DEPLOYMENT"

"%STREAMLIT%" run streamlit_app.py ^
    --server.port 8501 ^
    --server.headless true ^
    --browser.gatherUsageStats false ^
    --theme.base dark

echo.
echo  SafeGuard AI stopped. Press any key to close.
pause >nul
