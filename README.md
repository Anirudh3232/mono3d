# Mono3D - Sketch to 3D Model Converter

Mono3D is a web application that converts 2D sketches into 3D models using advanced AI technology. The project consists of a Next.js frontend and a Python Flask backend with TripoSR integration.

## Prerequisites

- Python 3.8 or higher
- Node.js 16 or higher
- npm (comes with Node.js)
- Git
- CUDA-compatible GPU (recommended for optimal performance)

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

1. **Check Requirements First:**
   ```bash
   cd backend
   python check_requirements.py
   ```

2. **Create and activate virtual environment:**
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # Linux/Mac
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   # For full installation (recommended)
   pip install -r requirements.txt
   
   # For minimal installation (CPU-only, basic features)
   pip install -r requirements-minimal.txt
   
   # For development (includes testing tools)
   pip install -r requirements-dev.txt
   ```

4. **Start the backend server:**
   ```bash
   python service.py
   ```
   The backend will run on http://localhost:5000

### Frontend Setup

1. **Install dependencies:**
   ```bash
   cd frontend
   npm install
   ```

2. **Start the development server:**
   ```bash
   npm run dev
   ```
   The frontend will run on http://localhost:3000

## Requirements Files

The backend includes multiple requirements files for different use cases:

- **`requirements.txt`** - Full installation with all features
- **`requirements-minimal.txt`** - CPU-only installation with basic features
- **`requirements-dev.txt`** - Development tools and testing dependencies

## Running the Application

1. **Start the backend server (from the backend directory):**
   ```bash
   python service.py
   ```
   The backend will run on http://localhost:5000

2. **Start the frontend server (from the frontend directory):**
   ```bash
   npm run dev
   ```
   The frontend will run on http://localhost:3000

3. **Access the application:**
   Open http://localhost:3000 in your browser

## API Endpoints

### Backend API
- `POST /generate` - Convert sketch to 3D model
- `GET /health` - Health check endpoint
- `GET /profiles` - Get available optimization profiles
- `GET /recommend` - Get recommended profile based on system specs
- `GET /test` - Test endpoint

## Project Structure

```
Mono3D/
├── frontend/           # Next.js frontend application
│   ├── src/
│   │   ├── app/       # Next.js app directory
│   │   ├── components/ # React components
│   │   └── types/     # TypeScript type definitions
│   ├── public/        # Static assets
│   └── package.json   # Node.js dependencies
│
├── backend/           # Python Flask backend
│   ├── service.py     # Main Flask application
│   ├── requirements.txt # Full dependencies
│   ├── requirements-minimal.txt # Minimal dependencies
│   ├── requirements-dev.txt # Development dependencies
│   ├── check_requirements.py # Requirements checker
│   ├── optimization_config.py # Performance profiles
│   └── TripoSR-main/ # TripoSR model integration
│
├── setup.bat         # Windows setup script
├── setup.sh          # Unix/Linux/Mac setup script
└── README.md         # This file
```

## Troubleshooting

### Common Issues

1. **CUDA not available:**
   - Install CUDA toolkit and cuDNN
   - Or use CPU-only installation: `pip install -r requirements-minimal.txt`

2. **Memory issues:**
   - Use a lower optimization profile
   - Reduce batch size in service.py
   - Close other applications to free memory

3. **Missing dependencies:**
   - Run `python check_requirements.py` to identify issues
   - Install missing packages manually if needed

4. **Model loading errors:**
   - Ensure TripoSR checkpoints are in the correct location
   - Check internet connection for model downloads

### Performance Optimization

The service includes multiple optimization profiles:
- **ultra_fast**: Minimal CPU usage for rapid prototyping
- **fast**: Balanced performance for general use
- **standard**: Optimized default settings
- **high_quality**: High quality output with moderate CPU usage
- **maximum_quality**: Maximum quality output (CPU intensive)

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