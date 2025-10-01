# Stage 1: Base environment build
FROM ubuntu:22.04 as builder

# Disable interactive prompts and configure time zone
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC

# Install apt-get dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates wget git openssh-client && \
    rm -rf /var/lib/apt/lists/*

# Install Miniconda
ARG MINICONDA_VERSION=py310_25.1.1-2
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-${MINICONDA_VERSION}-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh

# Set conda environment
ENV PATH=/opt/conda/bin:$PATH

# Stage 2: App deployment
FROM ubuntu:22.04

# Copy necessary content from the stage 1 (builder)
COPY --from=builder /opt/conda /opt/conda

# Set environment variables
ENV PATH=/opt/conda/bin:$PATH \
    LANG=C.UTF-8 \
    PYTHONUNBUFFERED=1

# Git clone the repository
RUN git clone --depth 1 https://github.com/chairc/Integrated-Design-Diffusion-Model.git /app

# Set working directory
WORKDIR /app

# Install (conda + pip)
# If pip is not available, use conda
RUN conda install -y pytorch=1.13.0 torchvision=0.14.0 torchaudio=0.13.0 pytorch-cuda=11.8 -c pytorch -c nvidia && \
    conda clean -ya

# If conda is not available, use pip
# RUN pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 -f https://download.pytorch.org/whl/torch_stable.html

RUN cd /app/Integrated-Design-Diffusion-Model && \
    pip install -r requirements.txt && \
    pip cache purge

# Verify installation
RUN python -c "import torch; print(f'PyTorch version: {torch.__version__}')" && \
    python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Set the default startup command
CMD ["/bin/bash"]