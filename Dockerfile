FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=/usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3-pip python3.10-dev \
        git wget build-essential ninja-build cmake \
        libgl1-mesa-glx libglib2.0-0 libegl1-mesa-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel

# Install PyTorch 2.6.0 with CUDA 12.4
RUN pip3 install --no-cache-dir \
    torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu124

# Clone TRELLIS.2
RUN git clone -b main https://github.com/microsoft/TRELLIS.2.git --recursive /app/trellis2

WORKDIR /app/trellis2

# Install basic dependencies
RUN pip3 install --no-cache-dir \
    numpy \
    Pillow \
    opencv-python-headless \
    tqdm \
    einops \
    omegaconf \
    safetensors \
    transformers \
    diffusers \
    accelerate \
    huggingface-hub

# V100 (SM 7.0) doesn't support flash-attn (requires SM 8.0+)
# Use PyTorch native SDPA instead (no extra install needed)

# Install nvdiffrast (v0.4.0) with explicit CUDA arch list
ENV TORCH_CUDA_ARCH_LIST="7.0;8.0;8.6"
RUN cd /tmp && \
    git clone -b v0.4.0 https://github.com/NVlabs/nvdiffrast.git && \
    cd nvdiffrast && \
    pip3 install --no-cache-dir --no-build-isolation . && \
    cd / && rm -rf /tmp/nvdiffrast
# Install custom TRELLIS.2 packages with CUDA arch list
ENV TORCH_CUDA_ARCH_LIST="7.0;8.0;8.6"

# CuMesh (external repo) - cubvh needs --extended-lambda for device lambdas
ENV NVCC_PREPEND_FLAGS="--extended-lambda"
RUN cd /tmp && \
    git clone https://github.com/JeffreyXiang/CuMesh.git --recursive && \
    cd CuMesh && \
    pip3 install --no-cache-dir --no-build-isolation . && \
    cd / && rm -rf /tmp/CuMesh
ENV NVCC_PREPEND_FLAGS=""

# o-voxel (bundled in TRELLIS.2 repo)
RUN cd /app/trellis2/o-voxel && \
    pip3 install --no-cache-dir --no-build-isolation .

# FlexGEMM skipped to save disk (optional, not required for inference)

ENV TORCH_CUDA_ARCH_LIST=""

# Model will be downloaded at runtime to save disk during build
# ENV HF_HOME=/app/models

WORKDIR /app
RUN mkdir -p /opt/artifact

# V100 doesn't support flash-attn; use PyTorch native SDPA
ENV ATTN_BACKEND=sdpa

COPY generate.py /app/generate.py
ENTRYPOINT ["python3", "/app/generate.py"]
