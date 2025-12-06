<div align="center">

<p>
  <a href="https://github.com/chairc/Integrated-Design-Diffusion-Model">
    <img width="242" height="121" src="./assets/iddm-logo.png" alt="IDDM">
  </a>
</p>

# IDDM: Integrated Design Diffusion Model

[English](README.md) | 简体中文

[![IDDM](https://img.shields.io/badge/IDDM-Integrated%20Design%20Diffusion%20Model-blue.svg)](https://github.com/chairc/Integrated-Design-Diffusion-Model)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10866128.svg)](https://doi.org/10.5281/zenodo.10866128)
[![zread](https://img.shields.io/badge/Ask_Zread-_.svg?style=flat&color=00b0aa&labelColor=000000&logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB3aWR0aD0iMTYiIGhlaWdodD0iMTYiIHZpZXdCb3g9IjAgMCAxNiAxNiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTQuOTYxNTYgMS42MDAxSDIuMjQxNTZDMS44ODgxIDEuNjAwMSAxLjYwMTU2IDEuODg2NjQgMS42MDE1NiAyLjI0MDFWNC45NjAxQzEuNjAxNTYgNS4zMTM1NiAxLjg4ODEgNS42MDAxIDIuMjQxNTYgNS42MDAxSDQuOTYxNTZDNS4zMTUwMiA1LjYwMDEgNS42MDE1NiA1LjMxMzU2IDUuNjAxNTYgNC45NjAxVjIuMjQwMUM1LjYwMTU2IDEuODg2NjQgNS4zMTUwMiAxLjYwMDEgNC45NjE1NiAxLjYwMDFaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik00Ljk2MTU2IDEwLjM5OTlIMi4yNDE1NkMxLjg4ODEgMTAuMzk5OSAxLjYwMTU2IDEwLjY4NjQgMS42MDE1NiAxMS4wMzk5VjEzLjc1OTlDMS42MDE1NiAxNC4xMTM0IDEuODg4MSAxNC4zOTk5IDIuMjQxNTYgMTQuMzk5OUg0Ljk2MTU2QzUuMzE1MDIgMTQuMzk5OSA1LjYwMTU2IDE0LjExMzQgNS42MDE1NiAxMy43NTk5VjExLjAzOTlDNS42MDE1NiAxMC42ODY0IDUuMzE1MDIgMTAuMzk5OSA0Ljk2MTU2IDEwLjM5OTlaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik0xMy43NTg0IDEuNjAwMUgxMS4wMzg0QzEwLjY4NSAxLjYwMDEgMTAuMzk4NCAxLjg4NjY0IDEwLjM5ODQgMi4yNDAxVjQuOTYwMUMxMC4zOTg0IDUuMzEzNTYgMTAuNjg1IDUuNjAwMSAxMS4wMzg0IDUuNjAwMUgxMy43NTg0QzE0LjExMTkgNS42MDAxIDE0LjM5ODQgNS4zMTM1NiAxNC4zOTg0IDQuOTYwMVYyLjI0MDFDMTQuMzk4NCAxLjg4NjY0IDE0LjExMTkgMS42MDAxIDEzLjc1ODQgMS42MDAxWiIgZmlsbD0iI2ZmZiIvPgo8cGF0aCBkPSJNNCAxMkwxMiA0TDQgMTJaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik00IDEyTDEyIDQiIHN0cm9rZT0iI2ZmZiIgc3Ryb2tlLXdpZHRoPSIxLjUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPgo8L3N2Zz4K&logoColor=ffffff)](https://zread.ai/chairc/Integrated-Design-Diffusion-Model)

[![License](https://img.shields.io/badge/license-Apache_2.0-green)](./LICENSE)
[![Issues](https://img.shields.io/github/issues/chairc/Integrated-Design-Diffusion-Model.svg)](https://github.com/chairc/Integrated-Design-Diffusion-Model/issues)
[![Release](https://img.shields.io/github/v/release/chairc/Integrated-Design-Diffusion-Model)](https://github.com/chairc/Integrated-Design-Diffusion-Model/releases)
[![Downloads](https://img.shields.io/github/downloads/chairc/Integrated-Design-Diffusion-Model/total?color=3eb370)](https://img.shields.io/github/downloads/chairc/Integrated-Design-Diffusion-Model/total?color=3eb370)

[![Github star](https://img.shields.io/github/stars/chairc/Integrated-Design-Diffusion-Model.svg)](https://github.com/chairc/Integrated-Design-Diffusion-Model/stargazers)
[![Github fork](https://img.shields.io/github/forks/chairc/Integrated-Design-Diffusion-Model?color=eb6ea5)](https://github.com/chairc/Integrated-Design-Diffusion-Model/forks)
[![Gitee star](https://gitee.com/chairc/Integrated-Design-Diffusion-Model/badge/star.svg?theme=blue)](https://gitee.com/chairc/Integrated-Design-Diffusion-Model)
[![Gitee fork](https://gitee.com/chairc/Integrated-Design-Diffusion-Model/badge/fork.svg?theme=blue)](https://gitee.com/chairc/Integrated-Design-Diffusion-Model)
[![GitCode star](https://gitcode.com/chairc/Integrated-Design-Diffusion-Model/star/badge.svg)](https://gitcode.com/chairc/Integrated-Design-Diffusion-Model)

</div>


### 关于模型

该扩散模型包含经典的ldm、ddpm、ddim和plms等采样算法，来源于《**[High-Resolution Image Synthesis with Latent Diffusion Models](https://openaccess.thecvf.com/content/CVPR2022/html/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper)**》、《**[Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)**》、《**[Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502)**》和《**[Pseudo Numerical Methods for Diffusion Models on Manifolds](https://openreview.net/forum?id=PlKWVd2yBkY)**》等论文。

我们将此项目命名为IDDM: Integrated Design Diffusion Model，中文名为集成设计扩散模型。在此项目中进行模型复现、训练器和生成器编写、部分算法和网络结构的改进与优化，该仓库**持续维护**。

如果有任何问题，请先到此[**issue**](https://github.com/chairc/Integrated-Design-Diffusion-Model/issues/9)进行问题查询，若无法解决可以加入我们的QQ群：949120343、开启新issue提问或联系我的邮箱：chenyu1998424@gmail.com/chairc1998@163.com。**如果你认为我的项目有意思请给我点一颗⭐⭐⭐Star⭐⭐⭐吧。**

### 快速开始

#### 本地运行

使用`git clone `方法或直接下载本仓库代码`zip`文件，本地配置环境运行即可

```bash
git clone https://github.com/chairc/Integrated-Design-Diffusion-Model.git
cd Integrated-Design-Diffusion-Model
# 在虚拟环境中运行项目（推荐）
conda create -n iddm python=3.10
pip install -r requirements.txt
# 训练模型
cd iddm/tools
python train.py --xxx xxx # 用个人参数替换-xxx
# 生成图像
python generate.py --xxx xxx # 用个人参数替换-xxx
```

#### 安装

除了本地运行外，也可采用下列两种方式安装本代码

**方式1**：使用[pip](https://pypi.org/project/iddm/)安装（推荐）

```bash
pip install iddm
```

需要以下前置安装包

```yaml
coloredlogs==15.0.1
gradio>=5.31.0
matplotlib==3.7.1
numpy==1.25.0
Pillow==10.3.0
Requests>=2.32.4
scikit-image==0.22.0
torch_summary==1.4.5
tqdm==4.66.3
pytorch_fid==0.3.0
fastapi==0.115.6
tensorboard==2.19.0
tensorboardX==2.6.1
transformers==4.56.1

# 如果您想使用 flash attention，请安装 flash-attn
# 编译自己的环境：pip install flash-attn --no-build-isolation
# 或从GitHub下载flash-attn.whl 文件：https://github.com/Dao-AILab/flash-attention/releases/tag/v2.8.2
# 可选安装（默认不安装）
flash-attn==2.8.2

# 如果你想下载GPU版本请使用：pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 -f https://download.pytorch.org/whl/torch_stable.html
# 想了解更多信息请访问：https://pytorch.org/get-started/previous-versions/#linux-and-windows-25
# 更多版本请访问：https://pytorch.org/get-started/previous-versions
# 需要注意torch版本 >= 1.10.0
 # 更多信息：https://pytorch.org/get-started/locally/ （推荐）
torch>=1.10.0
torchvision>=0.10.0
```

**方式2**：仓库安装

```bash
git clone https://github.com/chairc/Integrated-Design-Diffusion-Model.git
cd Integrated-Design-Diffusion-Model
pip install . # 或者 python setup.py install
```



### 接下来要做

- [x] [2023-07-15] 增加多卡分布式训练
- [x] [2023-07-31] 增加cosine学习率优化
- [x] [2023-08-03] 增加DDIM采样方法
- [x] [2023-08-28] 云服务器快速部署和接口
- [x] [2023-09-16] 支持其它图像生成
- [x] [2023-11-09] 增加效果更优的U-Net网络模型
- [x] [2023-11-09] 支持更大尺寸的生成图像
- [x] [2023-12-06] 重构model整体结构
- [x] [2024-01-23] 增加可视化webui训练界面
- [x] [2024-02-18] 支持低分辨率生成图像进行超分辨率增强[~~超分模型效果待定~~]
- [x] [2024-03-12] 增加PLMS采样方法
- [x] [2024-05-06] 增加FID方法验证图像质量
- [x] [2024-06-11] 增加可视化webui生成界面
- [x] [2024-07-07] 支持自定义图像长宽输入
- [x] [2024-11-13] 增加生成图像Socket和网站服务部署
- [x] [2024-11-26] 增加PSNR和SSIM方法验证超分图像质量
- [x] [2024-12-10] 增加预训练模型下载
- [x] [2024-12-25] 重构训练器结构
- [x] [2025-03-08] 支持PyPI包下载
- [x] [2025-08-01] 增加LDM方法，**支持生成512*512图像**，使用Latent方式降低显存消耗
- [x] [2025-08-21] 支持flash-attn，快速并节省显存
- [x] [2025-10-01] 增加Docker部署文件
- [ ] [待定] 重构项目利用百度飞桨框架



### 指南

开发、使用前请认真阅读指南内容哦~

| 指南名称 |                文档                 |
|:----:|:---------------------------------:|
| 模型训练 |  [训练.md](docs/zh-Hans/02_训练.md)   |
| 模型生成 |  [生成.md](docs/zh-Hans/03_生成.md)   |
| 模型结果 |  [结果.md](docs/zh-Hans/04_结果.md)   |
| 模型评估 |  [评估.md](docs/zh-Hans/05_评估.md)   |
| 模型列表 | [模型列表.md](docs/zh-Hans/06_模型库.md) |



### 引用

如果在学术论文中使用该项目进行实验，在可能的情况下，请适当引用我们的项目，为此我们表示感谢。具体引用格式可访问[**此网站**](https://zenodo.org/records/10866128)。

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

**引用详情可以参考此处**：

![image-20241124174257466](assets/image-citation.png)



### 致谢

[@dome272](https://github.com/dome272/Diffusion-Models-pytorch)，[@donger](https://donger.me/)和[@JetBrains](https://www.jetbrains.com/)

### 赞助商

![JetBrains logo](assets/jetbrains.svg)

