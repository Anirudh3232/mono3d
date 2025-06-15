# Mono3D - Sketch to 3D Model Converter

Mono3D is a web application that converts 2D sketches into 3D models using advanced AI technology. The project consists of a Next.js frontend and a Python FastAPI backend.

## Prerequisites

- Python 3.8 or higher
- Node.js 16 or higher
- npm (comes with Node.js)
- Git

## Quick Setup

### Windows
```bash
# Run the setup script
setup.bat
```

### Linux/Mac
```bash
# Make the script executable
chmod +x setup.sh

# Run the setup script
./setup.sh
```

## Manual Setup

### Backend Setup
1. Create and activate virtual environment:
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

2. Install dependencies:
```bash
cd backend
pip install -r requirements.txt
```

3. Start the backend server:
```bash
uvicorn app.main:app --reload
```

### Frontend Setup
1. Install dependencies:
```bash
cd frontend
npm install
```

2. Start the development server:
```bash
npm run dev
```

## Running the Application

1. Start the backend server (from the backend directory):
```bash
uvicorn app.main:app --reload
```
The backend will run on http://localhost:8000

2. Start the frontend server (from the frontend directory):
```bash
npm run dev
```
The frontend will run on http://localhost:3000

## Project Structure

```
Mono3D/
├── frontend/           # Next.js frontend application
│   ├── app/           # Next.js app directory
│   ├── components/    # React components
│   ├── public/        # Static assets
│   └── styles/        # CSS styles
│
├── backend/           # Python FastAPI backend
│   ├── app/          # FastAPI application
│   ├── models/       # AI models and utilities
│   └── requirements.txt  # Python dependencies
│
├── setup.bat         # Windows setup script
├── setup.sh          # Unix/Linux/Mac setup script
└── README.md         # This file
```

## API Endpoints

### Backend API
- `POST /api/sketch-to-3d`: Convert sketch to 3D model
- `GET /api/health`: Health check endpoint

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

Anirudh Rao - [@Anirudh3232](https://github.com/Anirudh3232) 