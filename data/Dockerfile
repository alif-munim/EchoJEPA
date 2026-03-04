FROM python:3.10-slim

# 1. Install System Dependencies
# ffmpeg: for video encoding
# libsm6, libxext6: required by OpenCV
# libgdcm-tools: required for compressed DICOMs
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 2. Install Python Packages
RUN pip install --no-cache-dir \
    pydicom \
    numpy \
    opencv-python-headless \
    tqdm \
    boto3

# 3. Entrypoint
ENTRYPOINT ["python3"]
