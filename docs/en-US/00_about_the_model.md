### About the Model

This diffusion model is based on the classic LDM (Latent Diffusion Models), DDPM (Denoising Diffusion Probabilistic Models), DDIM (Denoising Diffusion Implicit Models) and PLMS (Pseudo Numerical Methods for Diffusion Models on Manifolds) presented in the papers "[High-Resolution Image Synthesis with Latent Diffusion Models](https://openaccess.thecvf.com/content/CVPR2022/html/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper)", "[Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)", "[Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502)" and "[Pseudo Numerical Methods for Diffusion Models on Manifolds](https://openreview.net/forum?id=PlKWVd2yBkY)".

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
│   ├── autoencoder
│   │   └── train.py
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
│   │   │   ├── vae
│   │   │   │   └── autoencoder.py
│   │   │   ├── base.py
│   │   │   ├── conditional.py
│   │   │   ├── cspdarkunet.py
│   │   │   ├── unet.py
│   │   │   ├── unet_cross_attn.py
│   │   │   ├── unet_flash_self_attn.py
│   │   │   ├── unet_slim.py
│   │   │   └── unetv2.py
│   │   ├── samples
│   │   │   ├── base.py
│   │   │   ├── ddim.py
│   │   │   ├── ddpm.py
│   │   │   ├── dpm2.py
│   │   │   ├── dpmpp.py
│   │   │   └── plms.py
│   │   └── trainers
│   │       ├── autoencoder.py
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
│       ├── loss.py
│       ├── lr_scheduler.py
│       ├── metrics.py
│       ├── processing.py
│       └── utils.py
├── results
├── test
│   ├── noising_test
│   │   ├── landscape
│   │   └── noise
│   ├── test_check.py
│   ├── test_generate.py
│   ├── test_initializer.py
│   └── test_module.py
├── webui
│   └──web.py
├── weights
├── Dockerfile
├── requirements.txt
└── setup.py
```
