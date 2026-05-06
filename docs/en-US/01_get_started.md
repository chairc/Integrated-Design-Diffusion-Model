### Get Started

#### Running  Locally

Use the `git clone` or directly download the `zip` file of this repository's code, and then configure the environment locally to run it.

```bash
git clone https://github.com/chairc/Integrated-Design-Diffusion-Model.git
cd Integrated-Design-Diffusion-Model
# Run the project in a virtual environment (recommended)
conda create -n iddm python=3.10
pip install -r requirements.txt
# Train the model
cd iddm/tools
python train.py --xxx xxx # Replace --xxx with your training parameters
# Generate images
python generate.py --xxx xxx # Replace --xxx with your generation parameters
```

#### Docker Running

Modification contents, please refer to the `docker_run.sh` script.

```bash
# Mount and build the entire project
bash docker_run.sh -a /path/to/project
# Mount and build specified paths
bash docker_run.sh \
    -d /path/to/datasets \
    -r /path/to/results  \
    -w /path/to/weights
```

`/path/to/project`, `/path/to/datasets`, `/path/to/results` and `/path/to/weights` are local paths, used to mount the full project, dataset, result and weight directories respectively.

#### Installation

> [!NOTE]
>
> In addition to running locally, there are also the following two approachs for installing this code.
>

**Approach 1**: Use [pip](https://pypi.org/project/iddm/) install (Recommend)

```bash
pip install iddm
```

The following  packages are required.

```yaml
coloredlogs==15.0.1
gradio==6.0.0
matplotlib==3.7.1
numpy==1.25.0
Pillow==12.2.0
Requests==2.32.4
scikit-image==0.22.0
torch_summary==1.4.5
tqdm==4.66.3
pytorch_fid==0.3.0
fastapi==0.115.6
tensorboard==2.19.0
tensorboardX==2.6.1
transformers==5.0.0

# If you want to use flash attention, please install flash-attn.
# Compile your own environment: pip install flash-attn --no-build-isolation
# or download flash-attn .whl file from github: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.8.2
# Optional installation (Not installed by default)
flash-attn==2.8.2

# If you want to download gpu version
# Please use: pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 -f https://download.pytorch.org/whl/torch_stable.html
# About more torch information please click: https://pytorch.org/get-started/previous-versions/#linux-and-windows-25
# More versions please click: https://pytorch.org/get-started/previous-versions
# [Note] torch versions must >= 1.9.0
# More info: https://pytorch.org/get-started/locally/ (recommended)
torch>=1.10.0
torchvision>=0.10.0
```

**Approach 2**：Repository Installation

```bash
git clone https://github.com/chairc/Integrated-Design-Diffusion-Model.git
cd Integrated-Design-Diffusion-Model
pip install . # Or python setup.py install
```

