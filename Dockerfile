FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create user
RUN useradd -m -u 1000 user
USER user
WORKDIR /app

# Copy requirements and install dependencies
# We use --extra-index-url for CPU-only PyTorch to minimize image size
COPY --chown=user requirements.txt /app/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Create cache directory and set permissions
RUN mkdir -p /home/user/.cache && chmod -R 777 /home/user/.cache

# Copy application files
COPY --chown=user . /app

# Expose port 7860 for Hugging Face Space
EXPOSE 7860

# Start app
CMD ["python", "app.py"]
