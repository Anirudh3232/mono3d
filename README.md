# Mono3D - Sketch to 3D Image Generator

Mono3D is a web application that converts 2D sketches into 3D images using advanced AI technology. The project consists of a Next.js frontend and a Python Flask backend with TripoSR integration.

## Prerequisites

- Python 3.8 or higher
- Node.js 16 or higher
- npm (comes with Node.js)
- Git
- CUDA-compatible GPU (recommended for optimal performance)

## Quick Setup







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

## TripoSR Integration

The project includes a local TripoSR implementation for generating 3D images from sketches:

### Features
- **Local TripoSR Model**: Uses the included TripoSR-main directory
- **Background Removal**: Automatic background removal using rembg
- **Image Preprocessing**: Proper image formatting for TripoSR
- **3D Rendering**: Generates a single high-quality 3D image
- **Memory Optimization**: Chunked processing for better memory usage
- **Error Handling**: Graceful fallbacks and comprehensive logging

### Output
The service generates:
- **Single 3D Image**: A high-quality 3D rendered image from the sketch
- **Direct Display**: Image is displayed directly in the UI
- **Download**: Image can be downloaded as PNG file

### Colab Testing
The service is optimized for Google Colab testing with:
- Automatic GPU detection and optimization
- Memory management for Colab's limited resources
- Comprehensive error handling for remote execution
- Logging for debugging in Colab environment

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
- `POST /generate` - Convert sketch to 3D image
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
│   ├── colab_test.py # Colab testing script
│   └── TripoSR-main/ # Local TripoSR implementation
│       ├── tsr/      # TripoSR source code
│       ├── examples/ # Example images
│       └── run.py    # TripoSR CLI
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
   - Ensure TripoSR-main directory is present
   - Check internet connection for model downloads
   - Verify Python path includes TripoSR-main directory

5. **Colab-specific issues:**
   - Ensure GPU runtime is enabled in Colab
   - Check that all dependencies are installed
   - Monitor GPU memory usage in Colab

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
