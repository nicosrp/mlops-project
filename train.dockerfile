FROM python:3.11-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc curl && \
    apt clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Google Cloud SDK for gsutil
RUN curl https://sdk.cloud.google.com | bash
ENV PATH="/root/google-cloud-sdk/bin:${PATH}"

COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml

# Upgrade pip first
RUN pip install --upgrade pip setuptools wheel

# Install NumPy 1.x first to avoid compatibility issues with PyTorch 2.0.1
RUN pip install --no-cache-dir "numpy<2.0"

# Install dependencies with increased timeout
RUN pip install --default-timeout=300 -r requirements.txt --no-cache-dir
RUN pip install --no-cache-dir google-cloud-storage gcsfs

# Verify critical packages are installed and compatible
RUN python -c "import tqdm; import numpy; import torch; print(f'tqdm: {tqdm.__version__}, numpy: {numpy.__version__}, torch: {torch.__version__}')"

COPY configs/ configs/
COPY src/ src/

# Create directories for data and models
RUN mkdir -p data/processed models
RUN mkdir -p models

# Set environment variables
ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python", "-u", "src/mlops_project/train.py"]
