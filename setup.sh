#!/bin/bash

echo "Setting up Mono3D project..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed"
    exit 1
fi

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "Error: Node.js is not installed"
    exit 1
fi

# Create virtual environment for backend
echo "Creating Python virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "Error: Failed to create virtual environment"
        exit 1
    fi
else
    echo "Virtual environment already exists"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
if [ $? -ne 0 ]; then
    echo "Error: Failed to activate virtual environment"
    exit 1
fi

# Install backend dependencies
echo "Installing backend dependencies..."
if [ ! -d "backend" ]; then
    echo "Error: Backend directory not found"
    exit 1
fi

cd backend
if [ ! -f "requirements.txt" ]; then
    echo "Error: requirements.txt not found in backend directory"
    cd ..
    exit 1
fi

pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "Error: Failed to install backend dependencies"
    cd ..
    exit 1
fi
cd ..

# Install frontend dependencies
echo "Installing frontend dependencies..."
if [ ! -d "frontend" ]; then
    echo "Error: Frontend directory not found"
    exit 1
fi

cd frontend
npm install
if [ $? -ne 0 ]; then
    echo "Error: Failed to install frontend dependencies"
    cd ..
    exit 1
fi
cd ..

echo
echo "Setup complete!"
echo
echo "To start the backend:"
echo "1. Activate virtual environment: source venv/bin/activate"
echo "2. Navigate to backend: cd backend"
echo "3. Run: uvicorn app.main:app --reload"
echo
echo "To start the frontend:"
echo "1. Navigate to frontend: cd frontend"
echo "2. Run: npm run dev"
echo
echo "Backend will run on http://localhost:8000"
echo "Frontend will run on http://localhost:3000" 