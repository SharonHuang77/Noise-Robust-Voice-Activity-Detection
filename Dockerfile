
# Start with a Python 3.12 image
FROM python:3.12.9-slim

# Set environment variables to prevent Python from writing .pyc files
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /app

# Install system audio dependencies (CRITICAL for librosa/soundfile)
# libsndfile1 is the engine for reading audio files
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python libraries
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your project files
COPY . .

# Give execution permissions to the whole scripts folder
RUN chmod +x scripts/*.sh

CMD ["bash"]
# Full pipeliene execution: indexing, generation, and noise addition
# CMD python src/01_indexing/make_indexes.py \
#     --librispeech_root data/raw/LibriSpeech \
#     --musan_root data/raw/musan \
#     --out_dir data/indexes \
#     --ls_splits train-clean-100 dev-clean test-clean test-other && \
#     ./scripts/02_generation.sh && \
#     ./scripts/03_add_noise.sh