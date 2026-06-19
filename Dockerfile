# =============================================================================
# AI Stroke Painter - Dockerfile
# =============================================================================
# Multi-stage build that produces a container with:
#   - Python 3.11 + opencv/numpy/pillow (for model/inference.py + converter)
#   - Node.js 20 (for serving the renderer preview)
#   - The full project source
#
# Build:
#   docker build -t ai-stroke-painter .
#
# Run (preview server on port 8000):
#   docker run -p 8000:8000 ai-stroke-painter
#
# Run inference inside the container:
#   docker run -it --rm -v $(pwd)/assets:/workspace/assets ai-stroke-painter \
#     python model/inference.py --image assets/sample_cat.jpg --mode lite
#
# =============================================================================

# ---- Stage 1: Python + Node.js runtime -------------------------------------
FROM python:3.11-slim

# Install system dependencies for OpenCV and Node.js
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        curl \
        ca-certificates \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install Python dependencies first (better layer caching)
# PyTorch is optional and large; install CPU-only version for RL mode.
RUN pip install --no-cache-dir \
        opencv-python-headless \
        numpy \
        pillow \
        gdown \
    && pip install --no-cache-dir \
        torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    || echo "[Dockerfile] PyTorch install skipped (network issue); lite mode still works"

# Copy project source
COPY . /workspace/

# Pre-download weights (best-effort; don't fail the build if GDrive blocks it)
RUN bash model/download_weights.sh || \
    echo "[Dockerfile] Weight download skipped; lite mode will be used"

# Expose the preview server port
EXPOSE 8000

# Default: serve the renderer preview
# Users can override the command to run inference/converter instead.
CMD ["python", "-m", "http.server", "8000", "--bind", "0.0.0.0"]
