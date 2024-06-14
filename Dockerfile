FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-devel

RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    libgl1-mesa-glx \
    libgtk2.0-dev \
    libopenexr-dev && \
    rm -rf /var/lib/apt/lists/*
RUN pip3 install PyEXR==0.3.9 opencv-python 
RUN pip3 install parmap
RUN pip3 install OpenEXR
RUN pip3 install timm
RUN pip3 install einops

# Run 
WORKDIR /codes

