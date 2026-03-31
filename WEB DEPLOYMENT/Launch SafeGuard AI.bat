@echo off
title SafeGuard AI v2 — Industrial Safety Monitor
color 03

echo.
echo  ╔══════════════════════════════════════════════════════════════════╗
echo  ║        SafeGuard AI  —  Industrial Safety Monitor  v2.0         ║
echo  ║      YOLOv11n  ·  PPE Detection  ·  Tool Abandonment FSM        ║
echo  ╚══════════════════════════════════════════════════════════════════╝
echo.

:: Use Python312 which has all required packages + CUDA
set PYTHON_DIR=C:\Users\kbhas\AppData\Local\Programs\Python\Python312
set PATH=%PYTHON_DIR%;%PYTHON_DIR%\Scripts;%PATH%

echo  Starting SafeGuard AI dashboard...
echo.
echo  ┌─────────────────────────────────────────┐
echo  │  URL :  http://localhost:8501            │
echo  │  Stop:  Close this window               │
echo  └─────────────────────────────────────────┘
echo.

:: Open browser after 4 seconds
timeout /t 4 /nobreak >nul
start "" "http://localhost:8501"

:: Launch Streamlit dashboard
cd /d "E:\4TH YEAR PROJECT\WEB DEPLOYMENT"
streamlit run streamlit_app.py ^
    --server.port 8501 ^
    --server.headless true ^
    --browser.gatherUsageStats false ^
    --theme.base dark

echo.
echo  Server stopped. Press any key to exit.
pause >nul
