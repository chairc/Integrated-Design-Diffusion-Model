# IDDM: Industrial Defect Diffusion Model

[中文文档](README_zh.md)

### About the Model

The diffusion model used in this project is based on the classic ddpm introduced in the paper "[Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)".

We named this project IDDM: Industrial Defect Diffusion Model. It aims to reproduce the model, write trainers and generators, and improve and optimize certain algorithms and network structures. This repository is **actively maintained**.

**Repository Structure**

```yaml
├── datasets
│   └── dataset_demo
│       ├── class_1
│       ├── class_2
│       └── class_3
├── model
│   ├── base.py
│   ├── ddim.py
│   ├── ddpm.py
│   ├── modules.py
│   └── network.py
├── results
├── test
│   ├── noising_test
│   │   ├── landscape
│   │   └── noise
│   └── test_module.py
├── tools
│   ├── deploy.py
│   ├── generate.py
│   └── train.py
├── utils
│   ├── initializer.py
│   ├── lr_scheduler.py
│   └── utils.py
└── weight
```

### Next Steps

- [x] 1. Implement cosine learning rate optimization. (2023-07-31)
- [ ] 2. Use a more advanced U-Net network model.
- [ ] 3. Generate larger-sized images.
- [x] 4. Implement multi-GPU distributed training. (2023-07-15)
- [x] 5. Enable fast deployment and API on cloud servers. (2023-08-28)
- [x] 6. Adding DDIM Sampling Method. (2023-08-03)
- [x] 7. Support other image generation. (2023-09-16)

### Training

#### Start Your First Training (Using cifar10 as an Example, Single GPU Mode)

1. **Import the Dataset**

   First, upload the dataset to the target folder `datasets`. After uploading, the folder structure (for example, under the `cifar10` folder, there are folders for each class; `class0` folder contains all images for class 0) should look like the following:

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

   Set the `--num_classes` parameter to `10` (this is the total number of your classes.

   Set any other custom parameters as needed. If the error `CUDA out of memory` is shown in your terminal, turn down `--batch_size` and `num_workers`.

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
   python train.py --sample ddpm --conditional True --run_name df --epochs 300 --batch_size 16 --image_size 64 --num_classes 10 --dataset_path /your/dataset/path --result_path /your/save/path
   ```
   **Unconditional Training Command**

   ```bash
   python train.py --sample ddpm --conditional False --run_name df --epochs 300 --batch_size 16 --image_size 64 --dataset_path /your/dataset/path --result_path /your/save/path
   ```
4. Wait for the training to complete.
5. If the training is interrupted due to any reason, you can resume it by setting `--resume` to `True` in the `train.py` file, specifying the epoch number where the interruption occurred, providing the folder name of the interrupted training (`run_name`), and running the file again. Alternatively, you can use the following command to resume the training:  
   **Conditional Resume Training Command**

   ```bash
   python train.py --resume True --start_epoch 10 --load_model_dir df --sample ddpm --conditional True --run_name df --epochs 300 --batch_size 16 --image_size 64 --num_classes 10 --dataset_path /your/dataset/path --result_path /your/save/path
   ```
   **Unconditional Resume Training Command**

   ```bash
   python train.py --resume True --start_epoch 10 --load_model_dir df --sample ddpm --conditional False --run_name df --epochs 300 --batch_size 16 --image_size 64 --dataset_path /your/dataset/path --result_path /your/save/path
   ```
#### Distributed Training

1. The basic configuration is similar to regular training, but note that enabling distributed training requires setting `--distributed` to `True`. To prevent arbitrary use of distributed training, we have several conditions for enabling distributed training, such as `args.distributed`, `torch.cuda.device_count() > 1`, and `torch.cuda.is_available()`.

2. Set the necessary parameters, such as `--main_gpu` and `--world_size`. `--main_gpu` is usually set to the main GPU, which is used for validation, testing, or saving weights, and it only needs to be run on a single GPU. The value of `world_size` corresponds to the actual number of GPUs or distributed nodes being used.

3. There are two methods for setting the parameters. One is to directly modify the `parser` in the `train.py` file under the condition `if __name__ == "__main__":`. The other is to run the following command in the console under the path `/your/path/Defect-Diffiusion-Model/tools`:

   **Conditional Distributed Training Command**

   ```bash
   python train.py --sample ddpm --conditional True --run_name df --epochs 300 --batch_size 16 --image_size 64 --num_classes 10 --dataset_path /your/dataset/path --result_path /your/save/path --distributed True --main_gpu 0 --world_size 2
   ```

   **Unconditional Distributed Training Command**

   ```bash
   python train.py --sample ddpm --conditional False --run_name df --epochs 300 --batch_size 16 --image_size 64 --dataset_path /your/dataset/path --result_path /your/save/path --distributed True --main_gpu 0 --world_size 2
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
| --run_name             |             | File name                       | str  | File name used to initialize the model and save information  |
| --epochs               |             | Total number of epochs          | int  | Total number of training epochs                              |
| --batch_size           |             | Training batch size             | int  | Size of each training batch                                  |
| --num_workers          |             | Number of loading processes     | int  | Number of subprocesses used for data loading. It consumes a large amount of CPU and memory but can speed up training |
| --image_size           |             | Input image size                | int  | Input image size. Adaptive input and output sizes            |
| --dataset_path         |             | Dataset path                    | str  | Path to the conditional dataset, such as CIFAR-10, with each class in a separate folder, or the path to the unconditional dataset with all images in one folder |
| --fp16                 |             | Half precision training         | bool | Enable half precision training. It effectively reduces GPU memory usage but may affect training accuracy and results |
| --optim                |             | Optimizer                       | str  | Optimizer selection. Currently supports Adam and AdamW       |
| --act | | Activation function | str | Activation function selection. Currently supports gelu, silu, relu, relu6 and lrelu |
| --lr                   |             | Learning rate                   | float | Initial learning rate. Currently only supports linear learning rate |
| --lr_func | | Learning rate schedule | str | Setting learning rate schedule, currently supporting linear, cosine, and warmup_cosine. |
| --result_path          |             | Save path                       | str  | Path to save the training results                            |
| --save_model_interval  |             | Save model after each training  | bool | Whether to save the model after each training iteration for model selection based on visualization |
| --start_model_interval |             | Start epoch for saving models   | int  | Start epoch for saving models. This option saves disk space. If not set, the default is -1. If set, it starts saving models from the specified epoch. It needs to be used with --save_model_interval |
| --vis                  |             | Visualize dataset information   | bool | Enable visualization of dataset information for model selection based on visualization |
| --num_vis | | Number of visualization images generated | int | Number of visualization images generated. If not filled, the default is the number of image classes |
| --resume               |             | Resume interrupted training     | bool | Set to "True" to resume interrupted training. Note: If the epoch number of interruption is outside the condition of --start_model_interval, it will not take effect. For example, if the start saving model time is 100 and the interruption number is 50, we cannot set any loading epoch points because we did not save the model. We save the xxx_last.pt file every training, so we need to use the last saved model for interrupted training |
| --start_epoch          |             | Epoch number of interruption    | int  | Epoch number where the training was interrupted              |
| --load_model_dir       |             | Folder name of the loaded model | str  | Folder name of the previously loaded model                   |
| --distributed         |          | Distributed training          | bool  | Enable distributed training                                 |
| --main_gpu            |          | Main GPU for distributed      | int   | Set the main GPU for distributed training                   |
| --world_size          |          | Number of distributed nodes    | int   | Number of distributed nodes, corresponds to the actual number of GPUs or distributed nodes being used |
| --num_classes          |      ✓      | Number of classes               | int  | Number of classes used for classification                    |
| --cfg_scale            |      ✓      | Classifier-free guidance weight | int  | Classifier-free guidance interpolation weight for better model generation effects |

### Generation

1. Open the `generate.py` file and locate the `--weight_path` parameter. Modify the path in the parameter to the path of your model weights, for example `/your/path/weight/model.pt`.

2. Set the necessary parameters such as `--conditional`, `--generate_name`, `--num_images`, `--num_classes`, `--class_name`, `--image_size`, `--result_path`, etc. If no parameters are set, the default settings will be used. There are two ways to set the parameters: one is to directly modify the `parser` in the `if __name__ == "__main__":` section of the `generate.py` file, and the other is to use the following commands in the console while in the `/your/path/Defect-Diffusion-Model/tools` directory:

   **Conditional Generation Command**

   ```bash
   python generate.py --conditional True --generate_name df --num_images 8 --num_classes 10 --class_name 0 --image_size 64 --weight_path /your/path/weight/model.pt
   ```

   **Unconditional Generation Command**

   ```bash
   python generate.py --conditional False --generate_name df --num_images 8 --image_size 64 --weight_path /your/path/weight/model.pt
   ```

3. Wait for the generation process to complete.

#### Generation Parameters

**Parameter Explanation**

| Parameter Name  | Conditional | Usage                           | Type | Description                                                  |
| --------------- | :---------: | ------------------------------- | :--: | ------------------------------------------------------------ |
| --conditional   |             | Enable conditional generation   | bool | If enabled, allows custom configuration, such as modifying classes or classifier-free guidance interpolation weights |
| --generate_name |             | File name                       | str  | File name to initialize the model for saving purposes        |
| --image_size    |             | Input image size                | int  | Size of input images, adaptive input/output size. if class name is -1 and conditional is True, the model would output one image per class. |
| --num_images    |             | Number of generated images      | int  | Number of images to generate                                 |
| --weight_path   |             | Path to model weights           | str  | Path to the model weights file, required for network generation |
| --result_path   |             | Save path                       | str  | Path to save the generated images                            |
| --sample        |             | Sampling method                 | str  | Set the sampling method type, currently supporting DDPM and DDIM. |
| --act           |             | Activation function             | str  | Activation function selection. Currently supports gelu, silu, relu, relu6 and lrelu. If you do not set the same activation function as the model, mosaic phenomenon will occur. |
| --num_classes   |      ✓      | Number of classes               | int  | Number of classes for classification                         |
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

### Deployment

To be continued.

### Acknowledgements

[@dome272](https://github.com/dome272/Diffusion-Models-pytorch)

[@OpenAi](https://github.com/openai/improved-diffusion)