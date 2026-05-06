# Copyright 2025 IDDM Authors

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Use NVIDIA CUDA base image for GPU support
FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04
# For China region
# FROM swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

# Disable interactive prompts and configure time zone
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC \
    LANG=C.UTF-8 \
    PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends python3 python3-pip git && \
    rm -rf /var/lib/apt/lists/* && \
    ln -sf /usr/bin/python3 /usr/bin/python

# Clone the repository
RUN git clone --depth 1 https://github.com/chairc/Integrated-Design-Diffusion-Model.git /app
# For China region
# RUN git clone --depth 1 https://gitee.com/chairc/Integrated-Design-Diffusion-Model.git /app

# Set working directory
WORKDIR /app

# Install PyTorch (compatible with CUDA 12.8)
RUN pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
# For China region
# RUN pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

# Install project dependencies
RUN pip install --no-cache-dir --trusted-host pypi.org --trusted-host files.pythonhosted.org --no-deps -r requirements.txt
# For China region
# RUN pip install --no-cache-dir -r requirements.txt --index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

# Verify installation
RUN python -c "import torch; print(f'PyTorch version: {torch.__version__}')" && \
    python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Set the default startup command
CMD ["/bin/bash"]