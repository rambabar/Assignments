# Dockerfile, Image, Container
# Dockerfile - Blueprint for building images
# Image - Image is template for running container
# Container - Contrainer is actual running process where we have our package project

# Build stage
FROM python:3.9-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update \
    && apt-get install -y build-essential

# Copy requirements file
COPY requirements.txt .

# Install project dependencies
RUN pip install --no-cache-dir --user -r requirements.txt

# Runtime stage
FROM python:3.9-slim

WORKDIR /app

# Copy only necessary files from the build stage
COPY --from=builder /root/.local /root/.local
COPY . .

# Update PATH to include user-installed packages
ENV PATH=/root/.local/bin:$PATH

# Set the environment variable for pip cache directory
ENV PIP_CACHE_DIR=/pip_cache

# Install runtime dependencies
RUN apt-get update \
    && apt-get install -y ffmpeg libsm6 libxext6 libgl1-mesa-dev \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# # Install FastAPI explicitly
# RUN pip install --no-cache-dir fastapi uvicorn

# # Define the command to run the code
# CMD [ "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000" ]

# Define the command to run the code
CMD [ "python", "main.py" ]

# docker build -t rambabar108/kornia_image_matching:v1.0 .
# docker run -p 8000:8000 rambabar108/kornia_image_matching:v1.0
# docker push rambabar108/kornia_image_matching:v1.0
