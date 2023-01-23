FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel

# Test Nvidia Unit
RUN nvidia-smi

# Install dependencies
WORKDIR /openjourney
COPY ./discord_bot ./discord_bot
COPY ./sd_pipeline ./sd_pipeline
COPY ./.env ./.env
COPY ./openjourney.py ./openjourney.py
COPY ./export_user_prompts.py ./export_user_prompts.py
COPY ./requirements.txt ./requirements.txt

RUN apt update && apt install -y ffmpeg git wget build-essential unzip && apt install -y --no-install-recommends \
    pkg-config \
    libglvnd0 \
    libgl1 \
    libglx0 \
    libegl1 \
    libgles2 \
    libglvnd-dev \
    libgl1-mesa-dev \
    libegl1-mesa-dev \
    libgles2-mesa-dev \
    cmake \
    curl \
    libsm6 \
    libxext6 \
    libxrender-dev && rm -rf /var/lib/apt/lists/*

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility,graphics
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV CUDA_HOME=/usr/local/cuda

# Install CodeFormer dependencies
RUN git clone https://github.com/sczhou/CodeFormer CodeFormer
WORKDIR /openjourney/CodeFormer
RUN pip install -r requirements.txt
RUN python basicsr/setup.py develop
# Just for right .pth file installation, nvm
WORKDIR /openjourney
RUN python CodeFormer/scripts/download_pretrained_models.py CodeFormer
RUN python CodeFormer/scripts/download_pretrained_models.py facelib
ENV CODEFORMER_PATH /openjourney/CodeFormer/inference_codeformer.py

# Install environment variables for cache
ENV HF_HOME /openjourney/.cache
ENV TRANSFORMERS_CACHE /openjourney/.cache
ENV UPLOAD_PATH /openjourney/uploads

# Install xformers
WORKDIR /openjourney
RUN pip install ninja==1.11.1
ENV MAX_JOBS=8
ENV TORCH_CUDA_ARCH_LIST="7.5;8.0"
RUN git clone https://github.com/facebookresearch/xformers/
WORKDIR /openjourney/xformers
RUN git submodule update --init --recursive -q
RUN pip install --no-deps --verbose -e .

# Install dependencies for OpenJourney
WORKDIR /openjourney
RUN pip install -r requirements.txt

# Run contrainer
CMD ["python", "openjourney.py"]
