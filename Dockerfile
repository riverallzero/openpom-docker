# 1. Base image: Ubuntu 22.04 + CUDA 11.7
FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04

# 2. System package update & installation
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# 3. Set work directory
WORKDIR /openpom

# 4. Copy local file to container's WORKDIR
COPY . /openpom

# 5. Install python package
RUN pip install torch==2.0.1
RUN pip3 install --upgrade pip \
    && pip3 install -r requirements.txt

# 6. Set bash
CMD ["bash"]
