@echo off
title SafeGuard AI — Industrial Safety Monitor
color 0B

echo.
echo  ============================================================
echo    SafeGuard AI — Industrial Safety Detection System
echo    Starting server, please wait...
echo  ============================================================
echo.

:: Use Python312 which has all required packages
set PYTHON_DIR=C:\Users\kbhas\AppData\Local\Programs\Python\Python312
set PATH=%PYTHON_DIR%;%PYTHON_DIR%\Scripts;%PATH%

:: Wait a moment, then open the browser automatically
timeout /t 3 /nobreak >nul
start "" "http://localhost:8501"

:: Launch Streamlit (this line keeps the window open)
cd /d "E:\4TH YEAR PROJECT\WEB DEPLOYMENT"
streamlit run streamlit_app.py --server.port 8501 --server.headless true

echo.
echo  Server stopped. Press any key to exit.
pause >nul
