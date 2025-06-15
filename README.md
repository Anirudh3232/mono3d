# Sketch-to-3D Project

A modern implementation of sketch-to-3D conversion using Stable Diffusion and TripoSR.

## Project Structure

```
Mono3d/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   ├── core/
│   │   ├── models/
│   │   └── utils/
│   ├── config/
│   ├── tests/
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   └── utils/
│   ├── public/
│   └── Dockerfile
└── docker-compose.yml
```

## Setup Instructions

### Prerequisites
- Python 3.8+
- Node.js 16+
- CUDA-compatible GPU (or cloud GPU access)
- Docker (optional)

### Backend Setup
1. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\activate   # Windows
```

2. Install dependencies:
```bash
cd backend
pip install -r requirements.txt
```

3. Configure environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

### Frontend Setup
1. Install dependencies:
```bash
cd frontend
npm install
```

2. Configure environment:
```bash
cp .env.example .env.local
# Edit .env.local with your configuration
```

## Development

### Running Locally
1. Start backend:
```bash
cd backend
uvicorn app.main:app --reload
```

2. Start frontend:
```bash
cd frontend
npm run dev
```

### Running with Docker
```bash
docker-compose up --build
```

## Memory Optimization
- Uses FP16 precision by default
- Implements gradient checkpointing
- Utilizes xFormers for attention optimization
- Implements CPU offloading for large models

## API Documentation
- Backend API docs available at `/docs` when running
- Frontend API client generated using OpenAPI

## License
MIT 