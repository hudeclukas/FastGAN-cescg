FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ARG UID
ARG GID
ARG DEBIAN_FRONTEND=noninteractive

RUN groupadd -g $GID -o cescg && useradd -ms /bin/bash -u $UID -g $GID cescg && usermod -a -G cescg cescg
RUN apt update && apt install -y python3  \
    apt-utils \
    build-essential \
    python3-pip  \
    libpq-dev  \
    libglib2.0-0  \
    libgl1-mesa-glx

RUN mkdir -p /home/cescg && chmod 777 /home/cescg

COPY requirements.txt /home/cescg/requirements.txt
RUN pip install -r /home/cescg/requirements.txt
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

WORKDIR /home/cescg