# ============================================================
# LongCat-Video Environment Dockerfile
# Only builds the runtime environment, models are mounted at runtime
# ============================================================

FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

# Avoid interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
# NOTE: 不使用 python3-pip，避免 apt pip 与后续升级 pip 的 shebang 冲突
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    git \
    wget \
    curl \
    ffmpeg \
    libsndfile1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ninja-build \
    build-essential \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Set python3.10 as default (使用 ln -sf 替代 update-alternatives，更可靠)
RUN ln -sf /usr/bin/python3.10 /usr/bin/python \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3-config

# Install pip via get-pip.py (避免 apt python3-pip 的 shebang 冲突问题)
RUN curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py \
    && python3 /tmp/get-pip.py \
    && rm /tmp/get-pip.py \
    && python3 -m pip install --upgrade pip setuptools wheel

# Verify Python and pip are working
RUN python --version && python3 --version && pip --version

# Set work directory
WORKDIR /app

# Install PyTorch with CUDA 12.4 (官方源，稳定可靠)
RUN pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu124

# 直接安装预编译 flash-attn 包（推荐，无需编译，10秒完成）
# 国内用户请使用下面的 Hugging Face 镜像链接替换 GitHub 链接
RUN pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# 国内镜像备选（GitHub 访问失败时使用）
# RUN pip install https://hf-mirror.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# Copy requirements files first for better layer caching
COPY requirements.txt .
COPY requirements_avatar.txt .

# Install Python dependencies
RUN pip install -r requirements.txt
RUN pip install -r requirements_avatar.txt

# Copy project source code
COPY longcat_video/ ./longcat_video/
COPY assets/ ./assets/
COPY run_demo_text_to_video.py .
COPY run_demo_image_to_video.py .
COPY run_demo_video_continuation.py .
COPY run_demo_long_video.py .
COPY run_demo_interactive_video.py .
COPY run_demo_avatar_single_audio_to_video.py .
COPY run_demo_avatar_multi_audio_to_video.py .
COPY run_streamlit.py .
COPY run_streamlit_avatar.py .

# Create output directory
RUN mkdir -p /app/outputs

# Default environment variables for distributed inference
ENV RANK=0
ENV LOCAL_RANK=0
ENV WORLD_SIZE=1
ENV MASTER_ADDR=127.0.0.1
ENV MASTER_PORT=29500
ENV NCCL_P2P_DISABLE=0

# Expose Streamlit port
EXPOSE 8501

# NOTE: 不要使用 ENTRYPOINT ["bash"]！
#   会导致 docker run image xxx 变成 bash xxx，
#   bash 把 xxx 当脚本解释 → cannot execute binary file
# 正确做法：ENTRYPOINT 留空，CMD 给 bash 即可
ENTRYPOINT []
CMD ["bash"]
