FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

# Install dependencies
RUN mkdir /openjourney
WORKDIR /openjourney
COPY ./discord_bot ./discord_bot
COPY ./sd_pipeline ./sd_pipeline
COPY ./.env ./.env
COPY ./openjourney.py ./openjourney.py
COPY ./export_user_prompts.py ./export_user_prompts.py
COPY ./requirements.txt ./requirements.txt

RUN apt update && apt install -y ffmpeg git wget unzip && rm -rf /var/lib/apt/lists/*
RUN git clone https://github.com/sczhou/CodeFormer CodeFormer

# Install CodeFormer dependencies
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

# Install cutlass
WORKDIR /openjourney
RUN apt update && apt install -y build-essential && rm -rf /var/lib/apt/lists/*
RUN git clone https://github.com/NVIDIA/cutlass
WORKDIR /openjourney/cutlass
ENV CUDACXX=/opt/conda/bin/nvcc
RUN mkdir build
WORKDIR /openjourney/cutlass/build
RUN cmake ..

# Install xformers
WORKDIR /openjourney
RUN git clone https://github.com/facebookresearch/xformers
WORKDIR /openjourney/xformers
RUN git submodule update --init --recursive
RUN pip install --verbose --no-deps -e .

# Install dependencies for OpenJourney
WORKDIR /openjourney
RUN pip install -r requirements.txt

# Run contrainer
CMD ["python", "openjourney.py"]
