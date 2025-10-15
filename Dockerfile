# 베이스 이미지
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel

# 필수 패키지 설치
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    git \
    nano \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libxcb-xinerama0-dev \
 && rm -rf /var/lib/apt/lists/*