# IDDM: Integrated Design Diffusion Model

<div align="left">
    <a href="https://github.com/chairc/Integrated-Design-Diffusion-Model" target="_blank">
        <img src="https://img.shields.io/badge/IDDM-Integrated Design Diffusion Model-blue.svg">
    </a>
    <a href="https://doi.org/10.5281/zenodo.10866128">
        <img src="https://zenodo.org/badge/DOI/10.5281/zenodo.10866128.svg" alt="DOI">
    </a>
    <a href="https://github.com/chairc/Integrated-Design-Diffusion-Model/blob/main/LICENSE" target="_blank">
        <img src="https://img.shields.io/github/license/chairc/Integrated-Design-Diffusion-Model" />
    </a>
    <a href="https://github.com/chairc/Integrated-Design-Diffusion-Model/issues">
        <img src="https://img.shields.io/github/issues/chairc/Integrated-Design-Diffusion-Model.svg" />
    </a>
    <a href="https://github.com/chairc/Integrated-Design-Diffusion-Model/releases" target="_blank">
        <img alt="GitHub release" src="https://img.shields.io/github/v/release/chairc/Integrated-Design-Diffusion-Model">
    </a>
    <a href="#" target="_blank">
        <img alt="GitHub all releases" src="https://img.shields.io/github/downloads/chairc/Integrated-Design-Diffusion-Model/total?color=3eb370">
    </a>
</div>

<div align="left">
    <a href="https://github.com/chairc/Integrated-Design-Diffusion-Model/stargazers">
        <img src="https://img.shields.io/github/stars/chairc/Integrated-Design-Diffusion-Model.svg" />
    </a>
    <a href="https://github.com/chairc/Integrated-Design-Diffusion-Model/forks" target="_blank">
        <img alt="GitHub forks" src="https://img.shields.io/github/forks/chairc/Integrated-Design-Diffusion-Model?color=eb6ea5">
    </a>
    <a href="https://gitee.com/chairc/Integrated-Design-Diffusion-Model">
        <img src="https://gitee.com/chairc/Integrated-Design-Diffusion-Model/badge/star.svg?theme=blue" />
    </a>
    <a href="https://gitee.com/chairc/Integrated-Design-Diffusion-Model">
        <img src="https://gitee.com/chairc/Integrated-Design-Diffusion-Model/badge/fork.svg?theme=blue" />
    </a>
    <a href="https://gitcode.com/chairc/Integrated-Design-Diffusion-Model">
        <img src="https://gitcode.com/chairc/Integrated-Design-Diffusion-Model/star/badge.svg" />
    </a>
</div>

[中文文档](README_zh.md)



### About the Model

This diffusion model is based on the classic DDPM (Denoising Diffusion Probabilistic Models), DDIM (Denoising Diffusion Implicit Models) and PLMS (Pseudo Numerical Methods for Diffusion Models on Manifolds) presented in the papers "[Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)", "[Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502)" and "[Pseudo Numerical Methods for Diffusion Models on Manifolds](https://openreview.net/forum?id=PlKWVd2yBkY)".

We named this project IDDM: Integrated Design Diffusion Model. It aims to reproduce the model, write trainers and generators, and improve and optimize certain algorithms and network structures. This repository is **actively maintained**.

If you have any questions, please check [**the existing issues**](https://github.com/chairc/Integrated-Design-Diffusion-Model/issues/9) first. If the issue persists, feel free to open a new one for assistance, or you can contact me via email at chenyu1998424@gmail.com or chairc1998@163.com. **If you think my project is interesting, please give me a ⭐⭐⭐Star⭐⭐⭐ :)**

#### Running Locally

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

#### Installation

In addition to running locally, there are also the following two approachs for installing this code.

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



### Next Steps

- [x] [2023-07-15] Adding implement multi-GPU distributed training.
- [x] [2023-07-31] Adding implement cosine learning rate optimization.
- [x] [2023-08-03] Adding DDIM Sampling Method.
- [x] [2023-08-28] Adding fast deployment and API on cloud servers.
- [x] [2023-09-16] Support other image generation.
- [x] [2023-11-09] Adding a more advanced U-Net network model.
- [x] [2023-11-09] Support generate larger-sized images.
- [x] [2023-12-06] Refactor the overall structure of the model.
- [x] [2024-01-23] Adding visual webui training interface.
- [x] [2024-02-18] Support low-resolution generated images for super-resolution enhancement.[~~Super resolution model, the effect is uncertain~~]
- [x] [2024-03-12] Adding PLMS Sampling Method.
- [x] [2024-05-06] Adding FID calculator to verify image quality.
- [x] [2024-06-11] Adding visual webui generate interface.
- [x] [2024-07-07] Support custom images length and width input.
- [x] [2024-11-13] Adding the deployment of image-generating Sockets and Web server.
- [x] [2024-11-26] Adding PSNR and SSIM calculators to verify super resolution image quality.
- [x] [2024-12-10] Adding pretrain model download.
- [x] [2024-12-25] Refactor the overall structure of the trainer.
- [x] [2025-03-08] Support PyPI install.
- [ ] [Maybe 2025-01-31] Adding the deployment of Docker and image.
- [ ] [To be determined] Refactor the project by Baidu PaddlePaddle.
- [ ] [To be determined] ~~Use Latent Diffusion and reduce GPU memory usage~~



### Guide

Please read the guide carefully before developing or using.

|    Guide Name    |                   Document                   |
| :--------------: | :------------------------------------------: |
|  Model Training  |   [Training.md](docs/en-US/02_training.md)   |
| Model Generation | [Generation.md](docs/en-US/03_generation.md) |
|  Model Results   |    [Results.md](docs/en-US/04_results.md)    |
| Model Evaluation | [Evaluation.md](docs/en-US/05_evaluation.md) |



### Citation

If this project is used for experiments in an academic paper, where possible please cite our project appropriately and we appreciate this. The specific citation format can be found at **[this website](https://zenodo.org/records/10866128)**.

```
@software{chen_2024_10866128,
  author       = {Chen Yu},
  title        = {IDDM: Integrated Design Diffusion Model},
  month        = mar,
  year         = 2024,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.10866128},
  url          = {https://doi.org/10.5281/zenodo.10866128}
}
```

**Citation detail**:

 ![image-20241124174339833](assets/image-citation.png)



### Acknowledgements

[@dome272](https://github.com/dome272/Diffusion-Models-pytorch) and [@JetBrains](https://www.jetbrains.com/)



### Sponsor

![JetBrains logo](assets/jetbrains.svg)
