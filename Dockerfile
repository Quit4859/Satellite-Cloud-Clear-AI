FROM python:3.10-slim

WORKDIR /app

# System dependencies for OpenCV and rasterio
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgdal-dev \
    gdal-bin \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /app/uploads /app/outputs /app/temp

EXPOSE 8000 8501

# Default: run the API server
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
