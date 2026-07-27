@echo off
cd /d %~dp0

if not exist .env (
    if exist .env.example (
        copy .env.example .env >nul
        echo [INFO] .env created from .env.example
    )
)

pip show fastapi >nul 2>&1
if errorlevel 1 (
    echo [INFO] Installing dependencies...
    pip install -r requirements.txt
)

echo [INFO] Starting backend server...
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
