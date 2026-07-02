# Satellite Cloud Clear AI

A production-ready AI system for cloud removal and surface reconstruction from LISS-IV satellite imagery using multi-temporal satellite observations, foundation models, and deep learning.

## Overview

This project leverages multi-temporal satellite imagery and pre-trained AI models to remove clouds from LISS-IV satellite images and reconstruct the underlying surface. Built for hackathon deployment with a full-stack architecture.

## Tech Stack

- **Backend**: Python, FastAPI
- **Frontend**: Streamlit
- **AI/ML**: PyTorch, Hugging Face Transformers
- **Geospatial**: Rasterio, Google Earth Engine
- **Image Processing**: OpenCV
- **Testing**: pytest

## Project Structure

```
Satellite-Cloud-Clear-AI/
├── backend/          # FastAPI backend services
├── frontend/         # Streamlit web interface
├── models/           # Model architectures and definitions
├── utils/            # Utility functions and helpers
├── data/             # Data storage and processing
├── weights/          # Pre-trained model weights
├── notebooks/        # Jupyter notebooks for exploration
├── docs/             # Documentation
└── tests/            # Test suite
```

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/Satellite-Cloud-Clear-AI.git
cd Satellite-Cloud-Clear-AI

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Running the Application

```bash
# Start the backend
cd backend
uvicorn main:app --reload

# Start the frontend (in a new terminal)
cd frontend
streamlit run app.py
```

## API Documentation

Once the backend is running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Environment Variables

Create a `.env` file in the root directory:

```env
GEE_PROJECT_ID=your-google-earth-engine-project-id
HUGGINGFACE_TOKEN=your-huggingface-token
MODEL_WEIGHTS_PATH=./weights
DATA_DIR=./data
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
