### About the Model

This diffusion model is based on the classic DDPM (Denoising Diffusion Probabilistic Models), DDIM (Denoising Diffusion Implicit Models) and PLMS (Pseudo Numerical Methods for Diffusion Models on Manifolds) presented in the papers "[Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)", "[Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502)" and "[Pseudo Numerical Methods for Diffusion Models on Manifolds](https://openreview.net/forum?id=PlKWVd2yBkY)".

We named this project IDDM: Integrated Design Diffusion Model. It aims to reproduce the model, write trainers and generators, and improve and optimize certain algorithms and network structures. This repository is **actively maintained**.

If you have any questions, please check [**the existing issues**](https://github.com/chairc/Integrated-Design-Diffusion-Model/issues/9) first. If the issue persists, feel free to open a new one for assistance, or you can contact me via email at chenyu1998424@gmail.com or chairc1998@163.com. **If you think my project is interesting, please give me a ⭐⭐⭐Star⭐⭐⭐ :)**

#### Repository Structure

```yaml
Integrated Design Diffusion Model
├── datasets
│   └── dataset_demo
│       ├── class_1
│       ├── class_2
│       └── class_3
├── deploy
│   ├── deploy_socket.py
│   └── deploy_server.py
├── iddm
│   ├── config
│   │   ├── choices.py
│   │   ├── model_list.py
│   │   ├── setting.py
│   │   └── version.py
│   ├── model
│   │   ├── modules
│   │   │   ├── activation.py
│   │   │   ├── attention.py
│   │   │   ├── block.py
│   │   │   ├── conv.py
│   │   │   ├── ema.py
│   │   │   └── module.py
│   │   ├── networks
│   │   │   ├── sr
│   │   │   │   └── srv1.py
│   │   │   ├── base.py
│   │   │   ├── cspdarkunet.py
│   │   │   └── unet.py
│   │   ├── samples
│   │   │   ├── base.py
│   │   │   ├── ddim.py
│   │   │   ├── ddpm.py
│   │   │   └── plms.py
│   │   └── trainers
│   │       ├── base.py
│   │       ├── dm.py
│   │       └── sr.py
│   ├── sr
│   │   ├── dataset.py
│   │   ├── demo.py
│   │   ├── interface.py
│   │   └── train.py
│   ├── tools
│   │   ├── FID_calculator.py
│   │   ├── FID_calculator_plus.py
│   │   ├── generate.py
│   │   └── train.py
│   └── utils
│       ├── check.py
│       ├── checkpoint.py
│       ├── dataset.py
│       ├── initializer.py
│       ├── logger.py
│       ├── lr_scheduler.py
│       ├── metrics.py
│       ├── processing.py
│       └── utils.py
├── results
├── test
│   ├── noising_test
│   │   ├── landscape
│   │   └── noise
│   └── test_module.py
├── webui
│   └──web.py
└── weights
```

#### Running  Locally

Use the `git clone` or directly download the `zip` file of this repository's code, and then configure the environment locally to run it.

```bash
git clone https://github.com/chairc/Integrated-Design-Diffusion-Model.git
cd Integrated-Design-Diffusion-Model
```

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
gradio==5.0.0
matplotlib==3.7.1
numpy==1.25.0
Pillow==10.3.0
Requests==2.32.0
scikit-image==0.22.0
torch_summary==1.4.5
tqdm==4.66.3
pytorch_fid==0.3.0
fastapi==0.115.6
tensorboardX==2.6.1

# If you want to download gpu version
# Please use: pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 -f https://download.pytorch.org/whl/torch_stable.html
# About more torch information please click: https://pytorch.org/get-started/previous-versions/#linux-and-windows-25
# More versions please click: https://pytorch.org/get-started/previous-versions
# [Note] torch versions must >= 1.9.0
torch>=1.9.0 # More info: https://pytorch.org/get-started/locally/ (recommended)
torchvision>=0.10.0 # More info: https://pytorch.org/get-started/locally/ (recommended)
```

**Approach 2**：Repository Installation

```bash
git clone https://github.com/chairc/Integrated-Design-Diffusion-Model.git
cd Integrated-Design-Diffusion-Model
pip install . # Or python setup.py install
```

