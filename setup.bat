@echo off
setlocal enabledelayedexpansion

echo Setting up Mono3D project...

:: Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    exit /b 1
)

:: Check if Node.js is installed
node --version >nul 2>&1
if errorlevel 1 (
    echo Error: Node.js is not installed or not in PATH
    exit /b 1
)

:: Create virtual environment for backend
echo Creating Python virtual environment...
if exist venv (
    echo Virtual environment already exists
) else (
    python -m venv venv
    if errorlevel 1 (
        echo Error: Failed to create virtual environment
        exit /b 1
    )
)

:: Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate
if errorlevel 1 (
    echo Error: Failed to activate virtual environment
    exit /b 1
)

:: Install backend dependencies
echo Installing backend dependencies...
if not exist backend (
    echo Error: Backend directory not found
    exit /b 1
)
cd backend
if not exist requirements.txt (
    echo Error: requirements.txt not found in backend directory
    cd ..
    exit /b 1
)
pip install -r requirements.txt
if errorlevel 1 (
    echo Error: Failed to install backend dependencies
    cd ..
    exit /b 1
)
cd ..

:: Install frontend dependencies
echo Installing frontend dependencies...
if not exist frontend (
    echo Error: Frontend directory not found
    exit /b 1
)
cd frontend
call npm install
if errorlevel 1 (
    echo Error: Failed to install frontend dependencies
    cd ..
    exit /b 1
)
cd ..

echo.
echo Setup complete!
echo.
echo To start the backend:
echo 1. Activate virtual environment: venv\Scripts\activate
echo 2. Navigate to backend: cd backend
echo 3. Run: uvicorn app.main:app --reload
echo.
echo To start the frontend:
echo 1. Navigate to frontend: cd frontend
echo 2. Run: npm run dev
echo.
echo Backend will run on http://localhost:8000
echo Frontend will run on http://localhost:3000

endlocal 