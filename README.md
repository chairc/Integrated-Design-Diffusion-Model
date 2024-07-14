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
</div>

[中文文档](README_zh.md)

### About the Model

This diffusion model is based on the classic DDPM (Denoising Diffusion Probabilistic Models), DDIM (Denoising Diffusion Implicit Models) and PLMS (Pseudo Numerical Methods for Diffusion Models on Manifolds) presented in the papers "[Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)", "[Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502)" and "[Pseudo Numerical Methods for Diffusion Models on Manifolds](https://openreview.net/forum?id=PlKWVd2yBkY)".

We named this project IDDM: Integrated Design Diffusion Model. It aims to reproduce the model, write trainers and generators, and improve and optimize certain algorithms and network structures. This repository is **actively maintained**.

If you have any questions, please check [**the existing issues**](https://github.com/chairc/Integrated-Design-Diffusion-Model/issues/9) first. If the issue persists, feel free to open a new one for assistance, or you can contact me via email at chenyu1998424@gmail.com or chairc1998@163.com.

**Repository Structure**

```yaml
Integrated Design Diffusion Model
├── config
│   ├── choices.py
│   ├── setting.py
│   └── version.py
├── datasets
│   └── dataset_demo
│       ├── class_1
│       ├── class_2
│       └── class_3
├── model
│   ├── modules
│   │   ├── activation.py
│   │   ├── attention.py
│   │   ├── block.py
│   │   ├── conv.py
│   │   ├── ema.py
│   │   └── module.py
│   ├── networks
│   │   ├── sr
│   │   │   └── srv1.py
│   │   ├── base.py
│   │   ├── cspdarkunet.py
│   │   └── unet.py
│   └── samples
│       ├── base.py
│       ├── ddim.py
│       ├── ddpm.py
│       └── plms.py
├── results
├── sr
│   ├── dataset.py
│   ├── demo.py
│   ├── interface.py
│   └── train.py
├── test
│   ├── noising_test
│   │   ├── landscape
│   │   └── noise
│   └── test_module.py
├── tools
│   ├── deploy.py
│   ├── FID_calculator.py
│   ├── FID_calculator_plus.py
│   ├── generate.py
│   └── train.py
├── utils
│   ├── check.py
│   ├── checkpoint.py
│   ├── dataset.py
│   ├── initializer.py
│   ├── logger.py
│   ├── lr_scheduler.py
│   └── utils.py
├── webui
│   └──web.py
└── weight
```

### Next Steps

- [x] 1. Implement cosine learning rate optimization. (2023-07-31)
- [x] 2. Use a more advanced U-Net network model. (2023-11-09)
- [x] 3. Generate larger-sized images. (2023-11-09)
- [x] 4. Implement multi-GPU distributed training. (2023-07-15)
- [x] 5. Enable fast deployment and API on cloud servers. (2023-08-28)
- [x] 6. Adding DDIM Sampling Method. (2023-08-03)
- [x] 7. Support other image generation. (2023-09-16)
- [x] 8. Low-resolution generated images for super-resolution enhancement.[~~Super resolution model, the effect is uncertain.~~] (2024-02-18)
- [ ] 9. Use Latent Diffusion and reduce GPU memory usage
- [x] 10. Reconstruct the overall structure of the model (2023-12-06)
- [x] 11. Write visual webui interface. (2024-01-23)
- [x] 12. Adding PLMS Sampling Method. (2024-03-12)
- [x] 13. Adding FID calculator to verify image quality. (2024-05-06)

### Training

#### Note

The training GPU implements environment for this README is as follows: models are trained and tested with the NVIDIA RTX 3060 GPU with 6GB memory, NVIDIA RTX 2080Ti GPU with 11GB memory and NVIDIA RTX 6000 (×2) GPU with 24GB (total 48GB, distributed training) memory. **The above GPUs can all be trained normally**.

#### Start Your First Training (Using cifar10 as an Example, Single GPU Mode)

1. **Import the Dataset**

   First, upload the dataset to the target folder `datasets` [**[issue]**](https://github.com/chairc/Integrated-Design-Diffusion-Model/issues/9#issuecomment-1882902085). After uploading, the folder structure (for example, under the `cifar10` folder, there are folders for each class; `class0` folder contains all images for class 0) should look like the following:

   ```yaml
    datasets
    └── cifar10
        ├── class0
        ├── class1
        ├── class2
        ├── class3
        ├── class4
        ├── class5
        ├── class6
        ├── class7
        ├── class8
        └── class9
   ```

   At this point, your pre-training setup is complete.

2. **Set Training Parameters**

   Open the `train.py` file and modify the `parser` parameters inside the `if __name__ == "__main__":` block.

   Set the `--conditional` parameter to `True` because it's a multi-class training, so this needs to be enabled. For single-class, you can either not enable it or enable it.

   Set the `--run_name` parameter to the desired file name you want to create, for example, `cifar_exp1`.

   Set the `--dataset_path` parameter to the file path on your local or remote server, such as `/your/local/or/remote/server/file/path/datasets/cifar10`.

   Set the `--result_path` parameter to the file path on your local or remote server where you want to save the results.

   Set the `--num_classes` parameter to `10` (this is the total number of your classes).

   Set any other custom parameters as needed. If the error `CUDA out of memory` is shown in your terminal, turn down `--batch_size` and `num_workers`.

   In the custom parameters, you can set different `--sample` such as `ddpm` or `ddim` , and set different training networks `--network` such as `unet` or `cspdarkunet`. Of course, activation function `--act`, optimizer `--optim`, automatic mixed precision training `--amp`, learning rate method `--lr_func` and other parameters can also be customized.

   For detailed commands, refer to the **Training Parameters** section.

3. **Wait for the Training Process**

   After clicking `run`, the project will create a `cifar_exp1` folder in the `results` folder. This folder will contain training log files, model training files, model EMA (Exponential Moving Average) files, model optimizer files, all files saved during the last training iteration, and generated images after evaluation.

4. **View the Results**

   You can find the training results in the `results/cifar_exp1` folder.



**↓↓↓↓↓↓↓↓↓↓The following is an explanation of various training methods and detailed training parameters↓↓↓↓↓↓↓↓↓↓**

#### Normal Training

1. Take the `landscape` dataset as an example and place the dataset files in the `datasets` folder. The overall path of the dataset should be `/your/path/datasets/landscape`, the images path should be `/your/path/datasets/landscape/images`, and the image files should be located at `/your/path/datasets/landscape/images/*.jpg`.

2. Open the `train.py` file and locate the `--dataset_path` parameter. Modify the path in the parameter to the overall dataset path, for example, `/your/path/datasets/landscape`.

3. Set the necessary parameters such as `--sample`, `--conditional`, `--run_name`, `--epochs`, `--batch_size`, `--image_size`, `--result_path`, etc. If no parameters are set, the default settings will be used. There are two ways to set the parameters: directly modify the `parser` in the `if __name__ == "__main__":` section of the `train.py` file (**WE RECOMMEND THIS WAY**), or run the following command in the terminal at the `/your/path/Defect-Diffusion-Model/tools` directory:  
   **Conditional Training Command**

   ```bash
   python train.py --sample ddpm --conditional --run_name df --epochs 300 --batch_size 16 --image_size 64 --num_classes 10 --dataset_path /your/dataset/path --result_path /your/save/path
   ```
   **Unconditional Training Command**

   ```bash
   python train.py --sample ddpm --run_name df --epochs 300 --batch_size 16 --image_size 64 --dataset_path /your/dataset/path --result_path /your/save/path
   ```
4. Wait for the training to complete.
5. If the training is interrupted due to any reason **[[issue]](https://github.com/chairc/Integrated-Design-Diffusion-Model/issues/9#issuecomment-1882912391)**, you can resume it by setting `--resume` to `True` in the `train.py` file, specifying the epoch number where the interruption occurred, providing the folder name of the interrupted training (`run_name`), and running the file again. Alternatively, you can use the following command to resume the training:  
   **Conditional Resume Training Command**

   ```bash
   # This is using --start_epoch, default use current epoch checkpoint
   python train.py --resume --start_epoch 10 --sample ddpm --conditional --run_name df --epochs 300 --batch_size 16 --image_size 64 --num_classes 10 --dataset_path /your/dataset/path --result_path /your/save/path
   ```
   
   ```bash
   # This is not using --start_epoch, default use last checkpoint 
   python train.py --resume --sample ddpm --conditional --run_name df --epochs 300 --batch_size 16 --image_size 64 --num_classes 10 --dataset_path /your/dataset/path --result_path /your/save/path
   ```
   **Unconditional Resume Training Command**

   ```bash
   # This is using --start_epoch, default use current epoch checkpoint
   python train.py --resume --start_epoch 10 --sample ddpm --run_name df --epochs 300 --batch_size 16 --image_size 64 --dataset_path /your/dataset/path --result_path /your/save/path
   ```
   
   ```bash
   # This is not using --start_epoch, default use last checkpoint 
   python train.py --resume --sample ddpm --run_name df --epochs 300 --batch_size 16 --image_size 64 --dataset_path /your/dataset/path --result_path /your/save/path
   ```
6. The pretrained models are released with every major [Release](https://github.com/chairc/Integrated-Design-Diffusion-Model/releases), so please stay updated. To use a pretrained model [**[issue]**](https://github.com/chairc/Integrated-Design-Diffusion-Model/issues/9#issuecomment-1886403967), download the model corresponding to parameters such as `network`, `image_size`, `act`, etc., and save it to any local folder. Adjust the `--pretrain` and `--pretrain_path` in the `train.py` file accordingly. You can also use the following command for training with a pretrained model:

   **Command for conditional training with a pretrained model**

   ```bash
   python train.py --pretrain --pretrain_path /your/pretrain/path/model.pt --sample ddpm --conditional --run_name df --epochs 300 --batch_size 16 --image_size 64 --dataset_path /your/dataset/path --result_path /your/save/path
   ```

   **Command for unconditional training with a pretrained model**

   ```bash
   python train.py --pretrain --pretrain_path /your/pretrain/path/model.pt --sample ddpm --run_name df --epochs 300 --batch_size 16 --image_size 64 --dataset_path /your/dataset/path --result_path /your/save/path
   ```

#### Distributed Training

1. The basic configuration is similar to regular training, but note that enabling distributed training requires setting `--distributed`. To prevent arbitrary use of distributed training, we have several conditions for enabling distributed training, such as `args.distributed`, `torch.cuda.device_count() > 1`, and `torch.cuda.is_available()`.

2. Set the necessary parameters, such as `--main_gpu` and `--world_size`. `--main_gpu` is usually set to the main GPU, which is used for validation, testing, or saving weights, and it only needs to be run on a single GPU. The value of `world_size` corresponds to the actual number of GPUs or distributed nodes being used.

3. There are two methods for setting the parameters. One is to directly modify the `parser` in the `train.py` file under the condition `if __name__ == "__main__":`. The other is to run the following command in the console under the path `/your/path/Defect-Diffiusion-Model/tools`:

   **Conditional Distributed Training Command**

   ```bash
   python train.py --sample ddpm --conditional --run_name df --epochs 300 --batch_size 16 --image_size 64 --num_classes 10 --dataset_path /your/dataset/path --result_path /your/save/path --distributed --main_gpu 0 --world_size 2
   ```

   **Unconditional Distributed Training Command**

   ```bash
   python train.py --sample ddpm --run_name df --epochs 300 --batch_size 16 --image_size 64 --dataset_path /your/dataset/path --result_path /your/save/path --distributed --main_gpu 0 --world_size 2
   ```

4. Wait for the training to complete. Interrupt recovery is the same as basic training.

![IDDM Distributed Training](assets/IDDM_training.png)

#### Training Parameters

**Parameter Explanation**

| Parameter Name         | Conditional | Usage                           | Type | Description                                                  |
| ---------------------- | :---------: | ------------------------------- | :--: | ------------------------------------------------------------ |
| --seed |  | Initialize Seed | int | Set the seed for reproducible image generation from the network |
| --conditional          |             | Enable conditional training     | bool | Enable to modify custom configurations, such as modifying the number of classes and classifier-free guidance interpolation weights |
| --sample | | Sampling method | str | Set the sampling method type, currently supporting DDPM and DDIM. |
| --network | | Training network | str | Set the training network, currently supporting UNet, CSPDarkUNet. |
| --run_name             |             | File name                       | str  | File name used to initialize the model and save information  |
| --epochs               |             | Total number of epochs          | int  | Total number of training epochs                              |
| --batch_size           |             | Training batch size             | int  | Size of each training batch                                  |
| --num_workers          |             | Number of loading processes     | int  | Number of subprocesses used for data loading. It consumes a large amount of CPU and memory but can speed up training |
| --image_size           |             | Input image size                | int  | Input image size. Adaptive input and output sizes            |
| --dataset_path         |             | Dataset path                    | str  | Path to the conditional dataset, such as CIFAR-10, with each class in a separate folder, or the path to the unconditional dataset with all images in one folder |
| --amp                 |             | Automatic mixed precision training | bool | Enable automatic mixed precision training. It effectively reduces GPU memory usage but may affect training accuracy and results |
| --optim                |             | Optimizer                       | str  | Optimizer selection. Currently supports Adam and AdamW       |
| --loss                 |             | Loss function                   | str  | Loss selection. Currently supports MSELoss, L1Loss, HuberLoss and SmoothL1Loss             |
| --act | | Activation function | str | Activation function selection. Currently supports gelu, silu, relu, relu6 and lrelu |
| --lr                   |             | Learning rate                   | float | Initial learning rate. |
| --lr_func | | Learning rate schedule | str | Setting learning rate schedule, currently supporting linear, cosine, and warmup_cosine. |
| --result_path          |             | Save path                       | str  | Path to save the training results                            |
| --save_model_interval  |             | Save model after in training | bool | Whether to save the model after each training iteration for model selection based on visualization. If false, the model only save the last one |
| --save_model_interval_epochs | | Save the model interval | int | Save model interval and save it every X epochs |
| --start_model_interval |             | Start epoch for saving models   | int  | Start epoch for saving models. This option saves disk space. If not set, the default is -1. If set, it starts saving models from the specified epoch. It needs to be used with --save_model_interval |
| --vis                  |             | Visualize dataset information   | bool | Enable visualization of dataset information for model selection based on visualization |
| --num_vis | | Number of visualization images generated | int | Number of visualization images generated. If not filled, the default is the number of image classes |
| --image_format | | Generated image format in training | str | Generated image format in training, recommend to use png for better generation quality |
| --noise_schedule | | Noise schedule | str | This method is a model noise adding method |
| --resume               |             | Resume interrupted training     | bool | Set to "True" to resume interrupted training. Note: If the epoch number of interruption is outside the condition of --start_model_interval, it will not take effect. For example, if the start saving model time is 100 and the interruption number is 50, we cannot set any loading epoch points because we did not save the model. We save the xxx_last.pt file every training, so we need to use the last saved model for interrupted training |
| --start_epoch          |             | Epoch number of interruption    | int  | Epoch number where the training was interrupted, the model will load current checkpoint |
| --pretrain | | Enable use pretrain model | bool | Enable use pretrain mode to load checkpoint |
| --pretrain_path | | Pretrain model load path | str | Pretrain model load path |
| --use_gpu | | Set the use GPU | int | Set the use GPU in normal training, input is GPU's id |
| --distributed         |          | Distributed training          | bool  | Enable distributed training                                 |
| --main_gpu            |          | Main GPU for distributed      | int   | Set the main GPU for distributed training                   |
| --world_size          |          | Number of distributed nodes    | int   | Number of distributed nodes, corresponds to the actual number of GPUs or distributed nodes being used |
| --num_classes          |      ✓      | Number of classes               | int  | Number of classes used for classification **(No need to set for models after version 1.1.4)** |
| --cfg_scale            |      ✓      | Classifier-free guidance weight | int  | Classifier-free guidance interpolation weight for better model generation effects |



### Generation

1. Open the `generate.py` file and locate the `--weight_path` parameter. Modify the path in the parameter to the path of your model weights, for example `/your/path/weight/model.pt`.

2. Set the necessary parameters such as `--conditional`, `--generate_name`, `--num_images`, `--num_classes`, `--class_name`, `--image_size`, `--result_path`, etc. If no parameters are set, the default settings will be used. There are two ways to set the parameters: one is to directly modify the `parser` in the `if __name__ == "__main__":` section of the `generate.py` file, and the other is to use the following commands in the console while in the `/your/path/Defect-Diffusion-Model/tools` directory:

   **Conditional Generation Command (version > 1.1.1)**

   ```bash
   python generate.py --generate_name df --num_images 8 --class_name 0 --image_size 64 --weight_path /your/path/weight/model.pt --sample ddpm
   ```

   **Unconditional Generation Command (version > 1.1.1)**

   ```bash
   python generate.py --generate_name df --num_images 8 --image_size 64 --weight_path /your/path/weight/model.pt --sample ddpm
   ```

   **Conditional Generation Command (version <= 1.1.1)**

   ```bash
   python generate.py --conditional --generate_name df --num_images 8 --num_classes 10 --class_name 0 --image_size 64 --weight_path /your/path/weight/model.pt --sample ddpm --network unet --act gelu 
   ```

   **Unconditional Generation Command (version <= 1.1.1)**

   ```bash
   python generate.py --generate_name df --num_images 8 --image_size 64 --weight_path /your/path/weight/model.pt --sample ddpm --network unet --act gelu 
   ```

3. Wait for the generation process to complete.

#### Generation Parameters

**Parameter Explanation**

| Parameter Name  | Conditional | Usage                           | Type | Description                                                  |
| --------------- | :---------: | ------------------------------- | :--: | ------------------------------------------------------------ |
| --conditional   |             | Enable conditional generation   | bool | If enabled, allows custom configuration, such as modifying classes or classifier-free guidance interpolation weights |
| --generate_name |             | File name                       | str  | File name to initialize the model for saving purposes        |
| --image_size    |             | Input image size                | int  | Size of input images, adaptive input/output size. if class name is -1 and conditional is True, the model would output one image per class. |
| --image_format  |             | Generated image format          | str  | Generated image format, jpg/png/jpeg. Recommend to use png for better generation quality. |
| --num_images    |             | Number of generated images      | int  | Number of images to generate                                 |
| --weight_path   |             | Path to model weights           | str  | Path to the model weights file, required for network generation |
| --result_path   |             | Save path                       | str  | Path to save the generated images                            |
| --sample        |             | Sampling method                 | str  | Set the sampling method type, currently supporting DDPM and DDIM. **(No need to set for models after version 1.1.1)** |
| --network       |             | Training network                | str  | Set the training network, currently supporting UNet, CSPDarkUNet. **(No need to set for models after version 1.1.1)** |
| --act           |             | Activation function             | str  | Activation function selection. Currently supports gelu, silu, relu, relu6 and lrelu. If you do not set the same activation function as the model, mosaic phenomenon will occur. **(No need to set for models after version 1.1.1)** |
| --num_classes   |      ✓      | Number of classes               | int  | Number of classes for classification **(No need to set for models after version 1.1.1)** |
| --class_name    |      ✓      | Class name                      | int  | Index of the class to generate images. if class name is `-1`, the model would output one image per class. |
| --cfg_scale     |      ✓      | Classifier-free guidance weight | int  | Weight for classifier-free guidance interpolation, for better generation model performance |

### Result

We conducted training on the following four datasets using the `DDPM` sampler with an image size of `64*64`. we also enabled `conditional`, using the `gelu` activation function, `linear` learning function and  setting learning rate to `3e-4`. The datasets are `cifar10`, `NEUDET`, `NRSD-MN`, and `WOOD` in `300` epochs. The results are shown in the following figure:

#### cifar10

![cifar_244_ema](assets/cifar_244_ema.jpg)

![cifar_294_ema](assets/cifar_294_ema.jpg)

#### NEUDET

![neudet_290_ema](assets/neudet_290_ema.jpg)

![neudet_270_ema](assets/neudet_270_ema.jpg)

![neudet_276_ema](assets/neudet_276_ema.jpg)

![neudet_265_ema](assets/neudet_265_ema.jpg)

![neudet_240_ema](assets/neudet_240_ema.jpg)

![neudet_244_ema](assets/neudet_244_ema.jpg)

![neudet_298_ema](assets/neudet_298_ema.jpg)

#### NRSD

![nrsd_180_ema](assets/nrsd_180_ema.jpg)

![nrsd_188_ema](assets/nrsd_188_ema.jpg)

![nrsd_194_ema](assets/nrsd_194_ema.jpg)

![nrsd_203_ema](assets/nrsd_203_ema.jpg)

![nrsd_210_ema](assets/nrsd_210_ema.jpg)

![nrsd_217_ema](assets/nrsd_217_ema.jpg)

![nrsd_218_ema](assets/nrsd_218_ema.jpg)

![nrsd_248_ema](assets/nrsd_248_ema.jpg)

![nrsd_276_ema](assets/nrsd_276_ema.jpg)

![nrsd_285_ema](assets/nrsd_285_ema.jpg)

![nrsd_298_ema](assets/nrsd_298_ema.jpg)

#### WOOD

![wood_495](assets/wood_495.jpg)

#### Animate face (~~JUST FOR FUN~~)

![model_428_ema](assets/animate_face_428_ema.jpg)

![model_488_ema](assets/animate_face_488_ema.jpg)

![model_497_ema](assets/animate_face_497_ema.jpg)

![model_499_ema](assets/animate_face_499_ema.jpg)

![model_459_ema](assets/animate_face_459_ema.jpg)

#### Base on the 64×64 model to generate 160×160 (every size) images

Of course, based on the 64×64 U-Net model, we generate 160×160 `NEU-DET` images in the `generate.py` file (single output, each image occupies 21GB of GPU memory). **Attention this [[issues]](https://github.com/chairc/Integrated-Design-Diffusion-Model/issues/9#issuecomment-1886422210)**! If it's an image with defect textures where the features are not clear, generating a large size directly might not have these issues, such as in NRSD or NEU datasets. However, if the image contains a background with specific distinctive features, you may need to use super-resolution or resizing to increase the size, for example, in Cifar10, CelebA-HQ, etc. **If you really need large-sized images, you can directly train with large pixel images if there is enough GPU memory.** Detailed images are as follows:

![model_499_ema](assets/neu160_0.jpg)![model_499_ema](assets/neu160_1.jpg)![model_499_ema](assets/neu160_2.jpg)![model_499_ema](assets/neu160_3.jpg)![model_499_ema](assets/neu160_4.jpg)![model_499_ema](assets/neu160_5.jpg)


### Evaluation

1. During the data preparation stage, use `generate.py` to create the dataset. The amount and size of the generated dataset should be similar to the training set (**Note**: The training set required for evaluation should be resized to the size used during training, which is the `image_size`. For example, if the training set path is `/your/path/datasets/landscape` with an image size of **256**, and the generated set path is `/your/path/generate/landscape` with a size of 64, use the `resize` method to convert the images in the training set path to **64**. The new evaluation training set path will be `/your/new/path/datasets/landscape`).

2. Open the `FID_calculator.py` or `FID_calculator_plus.py` file for evaluation. `FID_calculator.py` is for **simple evaluation**; `FID_calculator_plus.py` is for **custom evaluation**, allowing various parameter settings.

3. If using `FID_calculator.py`, set `generated_image_folder` to `/your/path/generate/landscape` and `dataset_image_folder` to `/your/new/path/datasets/landscape`. **Right-click to run**.

4. If using `FID_calculator_plus.py`, set the necessary parameters such as `path`, `--batch_size`, `--num-workers`, `--dims`, `--save_stats`, and `--use_gpu`. If no parameters are set, the default settings will be used. There are two methods for setting parameters. One is to directly set the `parser` in the `if __name__ == "__main__":` block of the `FID_calculator_plus.py` file. The other is to enter the following command in the console under the `/your/path/Defect-Diffiusion-Model/tools` directory:  

   **For evaluation only**

   ```bash
   python FID_calculator_plus.py /your/path/generate/landscape /your/new/path/datasets/landscape --batch_size 8 --num-workers 2 --dims 2048 --use_gpu 0
   ```

   **To generate npz archives** (**generally not needed**)
   
   ```bash
   python FID_calculator_plus.py /your/input/path /your/output/path --save_stats
   ```

#### Evaluation Parameters

| **Parameter Name** | Usage                       | Parameter Type | Explanation                                                  |
| ------------------ | --------------------------- | :------------: | ------------------------------------------------------------ |
| path               | Path                        |      str       | Input two paths: the generated set path and the training set path in evaluation mode; input path and output path in npz mode |
| --batch_size       | Training batch size         |      int       | Size of each training batch                                  |
| --num_workers      | Number of loading processes |      int       | Number of subprocesses used for data loading. It consumes a large amount of CPU and memory but can speed up training |
| --dims             | Dimensions                  |      int       | The dimensions of the Inception features to use              |
| --save_stats       | Save stats                  |      bool      | Generate npz archives from the sample directory              |
| --use_gpu          | Specify GPU                 |      int       | Generally used to set the specific GPU for training, input the GPU number |

### About Citation

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

### Acknowledgements

**People**：

[@dome272](https://github.com/dome272/Diffusion-Models-pytorch)

**Organization**：

[@JetBrains](https://www.jetbrains.com/)

[@Python](https://www.python.org/)

[@Pytorch](https://pytorch.org/)

<img src="assets/Pycharm.png" alt="Pycharm" style="zoom:25%;" /><img src="assets/Python.png" alt="Python" style="zoom:25%;" /><img src="assets/pytorch.png" alt="pytorch" style="zoom:25%;" />