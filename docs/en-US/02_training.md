> [!NOTE]
>
> The training GPU implements environment for this README is as follows: 
>
> NVIDIA RTX 4070Ti: 16GB memory
>
> NVIDIA RTX 3060 (Laptop): 6GB memory
>
> NVIDIA RTX 2080Ti: 11GB memory
>
> NVIDIA RTX 6000 (×2): 22GB memory (total 44GB, distributed training)
>
> **The above GPUs can all be trained and tested normally**.



### 1. Diffusion Models Training

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

   Set any other custom parameters as needed. If the error `CUDA out of memory` is shown in your terminal, turn down `--batch_size` and `num_workers`.

   In the custom parameters, you can set different `--sample` such as `ddpm` or `ddim` , and set different training networks `--network` such as `unet` or `cspdarkunet`. Of course, activation function `--act`, optimizer `--optim`, automatic mixed precision training `--amp`, learning rate method `--lr_func` and other parameters can also be customized.

   For detailed commands, refer to the **Training Parameters** section.

3. **Wait for the Training Process**

   After clicking `run`, the project will create a `cifar_exp1` folder in the `results` folder. This folder will contain training log files, model training files, model EMA (Exponential Moving Average) files, model optimizer files, all files saved during the last training iteration, and generated images after evaluation.

4. **View the Results**

   You can find the training results in the `results/cifar_exp1` folder.



> [!NOTE]
>
> The following is an explanation of various training methods and detailed training parameters.
>



#### Normal Training

##### Command Training

1. Take the `landscape` dataset as an example and place the dataset files in the `datasets` folder. The overall path of the dataset should be `/your/path/datasets/landscape`, the images path should be `/your/path/datasets/landscape/images`, and the image files should be located at `/your/path/datasets/landscape/images/*.jpg`.

2. Open the `train.py` file and locate the `--dataset_path` parameter. Modify the path in the parameter to the overall dataset path, for example, `/your/path/datasets/landscape`.

3. Set the necessary parameters such as `--sample`, `--conditional`, `--run_name`, `--epochs`, `--batch_size`, `--image_size`, `--result_path`, etc. If no parameters are set, the default settings will be used. There are two ways to set the parameters: directly modify the `parser` in the `if __name__ == "__main__":` section of the `train.py` file (**WE RECOMMEND THIS WAY**), or run the following command in the terminal at the `/your/path/Defect-Diffusion-Model/iddm/tools` directory: 
   
   **Conditional Training Command**

   ```bash
   python train.py --sample ddpm --conditional --run_name df --epochs 300 --batch_size 16 --image_size 64 --dataset_path /your/dataset/path --result_path /your/save/path
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
   python train.py --resume --start_epoch 10 --sample ddpm --conditional --run_name df --epochs 300 --batch_size 16 --image_size 64 --dataset_path /your/dataset/path --result_path /your/save/path
   ```

   ```bash
   # This is not using --start_epoch, default use last checkpoint 
   python train.py --resume --sample ddpm --conditional --run_name df --epochs 300 --batch_size 16 --image_size 64 --dataset_path /your/dataset/path --result_path /your/save/path
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

7. During the training of a latent diffusion model, first set `latent` to `True`, `autoencoder_network` to the variational autoencoder model that provides encoding and decoding functions for this training session, and `autoencoder_ckpt` to the weight path of the current variational autoencoder. The remaining settings are the same as those for training an ordinary diffusion model. Alternatively, you can use the following commands for latent diffusion training:

   **Command for conditional training with latent diffusion model**

   ```bash
   python train.py --latent --autoencoder_network vae --autoencoder_ckpt /your/path/of/autoencoder/weight.pt --sample ddpm --conditional --run_name ldm --epochs 300 --batch_size 8 --image_size 64 --dataset_path /your/dataset/path --result_path /your/save/path
   ```

   **Command for unconditional training with latent diffusion model**

   ```bash
   python train.py --latent --autoencoder_network vae --autoencoder_ckpt /your/path/of/autoencoder/weight.pt --sample ddpm --run_name ldm --epochs 300 --batch_size 8 --image_size 64 --dataset_path /your/dataset/path --result_path /your/save/path
   ```

##### Python Training

```python
from iddm.model.trainers.dm import DMTrainer
from iddm.tools.train import init_train_args

# Approach 1
# Initialize arguments
args = init_train_args()
# Customize your parameters, or you can configure them by entering the init_train_args method
setattr(args, "conditional", True)  # True for conditional training, False for non-conditional training
setattr(args, "sample", "ddpm")  # Sampler
setattr(args, "network", "unet")  # Deep learning network
setattr(args, "epochs", 300)  # Number of iterations
setattr(args, "image_size", 64)  # Image size
setattr(args, "result_path", "/your/dataset/path/")  # Dataset path
setattr(args, "result_path", "/your/save/path/")  # Result path
setattr(args, "vis", True)  # Enable visualization
setattr(args, "latent", True) # Enable latent diffusion
setattr(args, "autoencoder_network", "vae") # VAE model
setattr(args, "autoencoder_ckpt", "/your/VAE/model/path/weight.pt") # VAE model weight path
# ...
# OR use args["parameter_name"] = "your setting"
# Start training
DMTrainer(args=args).train()

# Approach 2
args = init_train_args()
# Input args and update some params
DMTrainer(args=args, dataset_path="/your/dataset/path/").train()

# Approach 3
DMTrainer(
    conditional=True, sample="ddpm", dataset_path="/your/dataset/path/",
    network="unet", epochs=300, image_size=64, result_path="/your/save/path/",
    vis=True, latent=True, autoencoder_network="vae",
    autoencoder_ckpt="/your/VAE/model/path/weight.pt", # Any params...
).train()
```



#### Distributed Training

##### Command Training

1. The basic configuration is similar to regular training, but note that enabling distributed training requires setting `--distributed`. To prevent arbitrary use of distributed training, we have several conditions for enabling distributed training, such as `args.distributed`, `torch.cuda.device_count() > 1`, and `torch.cuda.is_available()`.

2. Set the necessary parameters, such as `--main_gpu` and `--world_size`. `--main_gpu` is usually set to the main GPU, which is used for validation, testing, or saving weights, and it only needs to be run on a single GPU. The value of `world_size` corresponds to the actual number of GPUs or distributed nodes being used.

3. There are two methods for setting the parameters. One is to directly modify the `parser` in the `train.py` file under the condition `if __name__ == "__main__":`. The other is to run the following command in the console under the path `/your/path/Defect-Diffiusion-Model/iddm/tools`:

   **Conditional Distributed Training Command**

   ```bash
   python train.py --sample ddpm --conditional --run_name df --epochs 300 --batch_size 16 --image_size 64 --dataset_path /your/dataset/path --result_path /your/save/path --distributed --main_gpu 0 --world_size 2
   ```

   **Unconditional Distributed Training Command**

   ```bash
   python train.py --sample ddpm --run_name df --epochs 300 --batch_size 16 --image_size 64 --dataset_path /your/dataset/path --result_path /your/save/path --distributed --main_gpu 0 --world_size 2
   ```

4. Wait for the training to complete. Interrupt recovery is the same as basic training.

![IDDM Distributed Training](../../assets/IDDM_training.png)

##### Python Training

```python
import torch
from torch import multiprocessing as mp
from iddm.model.trainers.dm import DMTrainer
from iddm.tools.train import init_train_args

# Approach 1
# Initialize arguments
args = init_train_args()
gpus = torch.cuda.device_count()
# Customize your parameters, or you can configure them by entering the init_train_args method
setattr(args, "distributed", True)  # Enable distributed training
setattr(args, "world_size", 2)  # Number of distributed nodes
setattr(args, "conditional", True)  # True for conditional training, False for non-conditional training
setattr(args, "sample", "ddpm")  # Sampler
setattr(args, "network", "unet")  # Deep learning network
setattr(args, "epochs", 300)  # Number of iterations
setattr(args, "image_size", 64)  # Image size
setattr(args, "result_path", "/your/dataset/path/")  # Dataset path
setattr(args, "result_path", "/your/save/path/")  # Result path
setattr(args, "vis", True)  # Enable visualization
setattr(args, "latent", True) # Enable latent diffusion
setattr(args, "autoencoder_network", "vae") # VAE model
setattr(args, "autoencoder_ckpt", "/your/VAE/model/path/weight.pt") # VAE model weight path
# ...
# OR use args["parameter_name"] = "your setting"
# Start training
mp.spawn(DMTrainer(args=args, dataset_path="/your/dataset/path/").train, nprocs=gpus)

# Approach 2
args = init_train_args()
# Input args and update some params
mp.spawn(DMTrainer(args=args, dataset_path="/your/dataset/path/").train, nprocs=gpus)

# Approach 3
mp.spawn(DMTrainer(
    conditional=True, sample="ddpm", dataset_path="/your/dataset/path/",
    network="unet", epochs=300, image_size=64, result_path="/your/save/path/",
    vis=True, latent=True, autoencoder_network="vae",
    autoencoder_ckpt="/your/VAE/model/path/weight.pt", # Any params...
).train, nprocs=gpus)
```



#### Training Parameters

**Parameter Explanation**

> [!WARNING]
>
> `--num_classes` do not need to set for models after **version 1.1.4**

| Parameter Name               | Conditional | Usage                                    | Type  | Description                                                  |
| ---------------------------- | :---------: | ---------------------------------------- | :---: | ------------------------------------------------------------ |
| --seed                       |             | Initialize Seed                          |  int  | Set the seed for reproducible image generation from the network. |
| --conditional                |             | Enable conditional training              | bool  | Enable to modify custom configurations, such as modifying the number of classes and classifier-free guidance interpolation weights. |
| --latent                     |             | Enable latent diffusion model            | bool  | If enabled, the model will use latent diffusion.             |
| --sample                     |             | Sampling method                          |  str  | Set the sampling method type, currently supporting DDPM and DDIM. |
| --network                    |             | Training network                         |  str  | Set the training network, currently supporting UNet, CSPDarkUNet. |
| --run_name                   |             | File name                                |  str  | File name used to initialize the model and save information. |
| --epochs                     |             | Total number of epochs                   |  int  | Total number of training epochs.                             |
| --batch_size                 |             | Training batch size                      |  int  | Size of each training batch.                                 |
| --num_workers                |             | Number of loading processes              |  int  | Number of subprocesses used for data loading. It consumes a large amount of CPU and memory but can speed up training. |
| --image_size                 |             | Input image size                         |  int  | Input image size. Adaptive input and output sizes.           |
| --dataset_path               |             | Dataset path                             |  str  | Path to the conditional dataset, such as CIFAR-10, with each class in a separate folder, or the path to the unconditional dataset with all images in one folder. |
| --amp                        |             | Automatic mixed precision training       | bool  | Enable automatic mixed precision training. It effectively reduces GPU memory usage but may affect training accuracy and results. |
| --optim                      |             | Optimizer                                |  str  | Optimizer selection. Currently supports Adam and AdamW.      |
| --loss                       |             | Loss function                            |  str  | Loss selection. Currently supports MSELoss, L1Loss, HuberLoss and SmoothL1Loss. |
| --act                        |             | Activation function                      |  str  | Activation function selection. Currently supports gelu, silu, relu, relu6 and lrelu. |
| --lr                         |             | Learning rate                            | float | Initial learning rate.                                       |
| --lr_func                    |             | Learning rate schedule                   |  str  | Setting learning rate schedule, currently supporting linear, cosine, and warmup_cosine. |
| --result_path                |             | Save path                                |  str  | Path to save the training results.                           |
| --save_model_interval        |             | Save model after in training             | bool  | Whether to save the model after each training iteration for model selection based on visualization. If false, the model only save the last one. |
| --save_model_interval_epochs |             | Save the model interval                  |  int  | Save model interval and save it every X epochs.              |
| --start_model_interval       |             | Start epoch for saving models            |  int  | Start epoch for saving models. This option saves disk space. If not set, the default is -1. If set, it starts saving models from the specified epoch. It needs to be used with --save_model_interval. |
| --vis                        |             | Visualize dataset information            | bool  | Enable visualization of dataset information for model selection based on visualization. |
| --num_vis                    |             | Number of visualization images generated |  int  | Number of visualization images generated. If not filled, the default is the number of image classes. |
| --image_format               |             | Generated image format in training       |  str  | Generated image format in training, recommend to use png for better generation quality. |
| --noise_schedule             |             | Noise schedule                           |  str  | This method is a model noise adding method.                  |
| --resume                     |             | Resume interrupted training              | bool  | Set to "True" to resume interrupted training. Note: If the epoch number of interruption is outside the condition of --start_model_interval, it will not take effect. For example, if the start saving model time is 100 and the interruption number is 50, we cannot set any loading epoch points because we did not save the model. We save the xxx_last.pt file every training, so we need to use the last saved model for interrupted training. |
| --start_epoch                |             | Epoch number of interruption             |  int  | Epoch number where the training was interrupted, the model will load current checkpoint. |
| --pretrain                   |             | Enable use pretrain model                | bool  | Enable use pretrain mode to load checkpoint.                 |
| --pretrain_path              |             | Pretrain model load path                 |  str  | Pretrain model load path.                                    |
| --use_gpu                    |             | Set the use GPU                          |  int  | Set the use GPU in normal training, input is GPU's id.       |
| --distributed                |             | Distributed training                     | bool  | Enable distributed training.                                 |
| --main_gpu                   |             | Main GPU for distributed                 |  int  | Set the main GPU for distributed training.                   |
| --world_size                 |             | Number of distributed nodes              |  int  | Number of distributed nodes, corresponds to the actual number of GPUs or distributed nodes being used. |
| --num_classes                |      ✓      | Number of classes                        |  int  | Number of classes used for classification **(No need to set for models after version 1.1.4)**. |
| --cfg_scale                  |      ✓      | Classifier-free guidance weight          |  int  | Classifier-free guidance interpolation weight for better model generation effects. |
| --autoencoder_network        |             | VAE model                                |  str  | VAE model support encode and decode.                         |
| --autoencoder_ckpt           |             | VAE model weight path                    |  str  | VAE model weight path.                                       |



### 2. Autoencoder Model Training

#### Before Training

1. **Import the Dataset**

   First, upload the dataset to the target folder `datasets`. After uploading, the folder structure (neu-det folder contains train and val; All the training images are stored in the images folder) should look like the following:

   ```yaml
   datasets
      └── neu-det
          │
          ├── train
          │   └── images
          │       ├── image_1.jpg
          │       ├── image_2.jpg
          │       └── ...
          │
          └── val
              └── images
                  ├── image_1.jpg
                  ├── image_2.jpg
                  └── ...
   ```

   At this point, your pre-training setup is complete.

2. **Set Training Parameters**

   Open the `/iddm/autoencoder/train.py` file and modify the `parser` parameters inside the `if __name__ == "__main__":` block.

   Set the `--run_name` parameter to the desired file name you want to create, for example, `neudet_autoencoder`.

   Set the `--images_size` parameter to the size of your image input (recommended default), e.g. '512'.

   Set the `--dataset_path` parameter to the file path on your local or remote server, such as `/your/local/or/remote/server/file/path/datasets/neudet`.

   Set the `--result_path` parameter to the file path on your local or remote server where you want to save the results.

   Set any other custom parameters as needed. If the error `CUDA out of memory` is shown in your terminal, turn down `--batch_size` and `num_workers`.

   In the custom parameters, you can set different training networks `--network` such as `vae`. Of course, activation function `--act`, optimizer `--optim`, automatic mixed precision training `--amp`, learning rate method `--lr_func` and other parameters can also be customized.

   For detailed commands, refer to the **Training Parameters** section.

3. **Wait for the Training Process**

   After clicking `run`, the project will create a `neudet_autoencoder` folder in the `results` folder. This folder will contain training log files, model training files, model EMA (Exponential Moving Average) files, model optimizer files, all files saved during the last training iteration, and generated images after evaluation.

4. **View the Results**

   You can find the training results in the `results/neudet_autoencoder` folder.




#### Normal Training

> [!NOTE]
>
> For detailed training, please refer to:**1. Diffusion Models Training - Normal Training**
>



#### Distributed Training

> [!NOTE]
>
> For detailed training, please refer to:**1. Diffusion Models Training - Distributed Training**



#### Training Parameters

**Parameter Explanation**


| Parameter Name               | Usage                              | Type  | Description                                                  |
| ---------------------------- | ---------------------------------- | :---: | ------------------------------------------------------------ |
| --seed                       | Initialize Seed                    |  int  | Set the seed for reproducible image generation from the network. |
| --network                    | Training network                   |  str  | Set the training network, currently supporting UNet, CSPDarkUNet. |
| --run_name                   | File name                          |  str  | File name used to initialize the model and save information. |
| --epochs                     | Total number of epochs             |  int  | Total number of training epochs.                             |
| --batch_size                 | Training batch size                |  int  | Size of each training batch.                                 |
| --num_workers                | Number of loading processes        |  int  | Number of subprocesses used for data loading. It consumes a large amount of CPU and memory but can speed up training. |
| --image_size                 | Input image size                   |  int  | Input image size. Adaptive input and output sizes.           |
| --latent_channels            | The latent space                   |  int  | The number of channels in the latent space.                  |
| --train_dataset_path         | Train Dataset path                 |  str  | Train Dataset path.                                          |
| --val_dataset_path           | Val dataset path                   |  str  | Val dataset path.                                            |
| --amp                        | Automatic mixed precision training | bool  | Enable automatic mixed precision training. It effectively reduces GPU memory usage but may affect training accuracy and results. |
| --optim                      | Optimizer                          |  str  | Optimizer selection. Currently supports Adam and AdamW.      |
| --loss                       | Loss function                      |  str  | Loss selection. Currently supports MSELoss, L1Loss, HuberLoss and SmoothL1Loss. |
| --act                        | Activation function                |  str  | Activation function selection. Currently supports gelu, silu, relu, relu6 and lrelu. |
| --lr                         | Learning rate                      | float | Initial learning rate.                                       |
| --lr_func                    | Learning rate schedule             |  str  | Setting learning rate schedule, currently supporting linear, cosine, and warmup_cosine. |
| --result_path                | Save path                          |  str  | Path to save the training results.                           |
| --save_model_interval        | Save model after in training       | bool  | Whether to save the model after each training iteration for model selection based on visualization. If false, the model only save the last one. |
| --save_model_interval_epochs | Save the model interval            |  int  | Save model interval and save it every X epochs.              |
| --start_model_interval       | Start epoch for saving models      |  int  | Start epoch for saving models. This option saves disk space. If not set, the default is -1. If set, it starts saving models from the specified epoch. It needs to be used with --save_model_interval. |
| --image_format               | Generated image format in training |  str  | Generated image format in training, recommend to use png for better generation quality. |
| --resume                     | Resume interrupted training        | bool  | Set to "True" to resume interrupted training. Note: If the epoch number of interruption is outside the condition of --start_model_interval, it will not take effect. For example, if the start saving model time is 100 and the interruption number is 50, we cannot set any loading epoch points because we did not save the model. We save the xxx_last.pt file every training, so we need to use the last saved model for interrupted training. |
| --start_epoch                | Epoch number of interruption       |  int  | Epoch number where the training was interrupted, the model will load current checkpoint. |
| --pretrain                   | Enable use pretrain model          | bool  | Enable use pretrain mode to load checkpoint.                 |
| --pretrain_path              | Pretrain model load path           |  str  | Pretrain model load path.                                    |
| --use_gpu                    | Set the use GPU                    |  int  | Set the use GPU in normal training, input is GPU's id.       |
| --distributed                | Distributed training               | bool  | Enable distributed training.                                 |
| --main_gpu                   | Main GPU for distributed           |  int  | Set the main GPU for distributed training.                   |
| --world_size                 | Number of distributed nodes        |  int  | Number of distributed nodes, corresponds to the actual number of GPUs or distributed nodes being used. |
