# IDDM: Industrial Defect Diffusion Model

[中文文档](README_zh.md)

### About the Model

The diffusion model used in this project is based on the classic ddpm introduced in the paper "[Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)".

We named this project IDDM: Industrial Defect Diffusion Model. It aims to reproduce the model, write trainers and generators, and improve and optimize certain algorithms and network structures. This repository is **actively maintained**.

**Repository Structure**

```yaml
├── datasets
├── model
│   ├── ddpm.py
│   ├── modules.py
│   └── network.py
├── results
├── test
│   ├── noising_test
│   │   ├── landscape
│   │   │   └── noising_test.jpg
│   │   └── noise
│   │       └── noise.jpg
│   └── test_module.py
├── tools
│   ├── generate.py
│   └── train.py
├── utils
│   ├── initializer.py
│   └── utils.py
└── weight
```

### Next Steps

- [ ] 1. Implement cosine learning rate optimization.
- [ ] 2. Use a more advanced U-Net network model.
- [ ] 3. Generate larger-sized images.
- [x] 4. Implement multi-GPU distributed training.
- [ ] 5. Enable fast deployment on cloud servers.

### Training

#### Normal Training

1. Take the `landscape` dataset as an example and place the dataset files in the `datasets` folder. The overall path of the dataset should be `/your/path/datasets/landscape`, and the image files should be located at `/your/path/datasets/landscape/*.jpg`.

2. Open the `train.py` file and locate the `--dataset_path` parameter. Modify the path in the parameter to the overall dataset path, for example, `/your/path/datasets/landscape`.

3. Set the necessary parameters such as `--conditional`, `--run_name`, `--epochs`, `--batch_size`, `--image_size`, `--result_path`, etc. If no parameters are set, the default settings will be used. There are two ways to set the parameters: directly modify the `parser` in the `if __name__ == "__main__":` section of the `train.py` file, or run the following command in the terminal at the `/your/path/Defect-Diffusion-Model/tools` directory:
   **Conditional Training Command**

   ```bash
   python train.py --conditional True --run_name 'df' --epochs 300 --batch_size 16 --image_size 64 --num_classes 10
   ```
   **Unconditional Training Command**

   ```bash
   python train.py --conditional False --run_name 'df' --epochs 300 --batch_size 16 --image_size 64
   ```
4. Wait for the training to complete.
5. If the training is interrupted due to any reason, you can resume it by setting `--resume` to `True` in the `train.py` file, specifying the epoch number where the interruption occurred, providing the folder name of the interrupted training (`run_name`), and running the file again. Alternatively, you can use the following command to resume the training:
   **Conditional Resume Training Command**

   ```bash
   python train.py --resume True --start_epoch 10 --load_model_dir 'df' --conditional True --run_name 'df' --epochs 300 --batch_size 16 --image_size 64 --num_classes 10
   ```
   **Unconditional Resume Training Command**

   ```bash
   python train.py --resume True --start_epoch 10 --load_model_dir 'df'  --conditional False --run_name 'df' --epochs 300 --batch_size 16 --image_size 64
   ```
#### Distributed Training

1. The basic configuration is similar to regular training, but note that enabling distributed training requires setting `--distributed` to `True`. To prevent arbitrary use of distributed training, we have several conditions for enabling distributed training, such as `args.distributed`, `torch.cuda.device_count() > 1`, and `torch.cuda.is_available()`.

2. Set the necessary parameters, such as `--main_gpu` and `--world_size`. `--main_gpu` is usually set to the main GPU, which is used for validation, testing, or saving weights, and it only needs to be run on a single GPU. The value of `world_size` corresponds to the actual number of GPUs or distributed nodes being used.

3. There are two methods for setting the parameters. One is to directly modify the `parser` in the `train.py` file under the condition `if __name__ == "__main__":`. The other is to run the following command in the console under the path `/your/path/Defect-Diffiusion-Model/tools`:

**Conditional Distributed Training Command**

   ```bash
   python train.py --conditional True --run_name 'df' --epochs 300 --batch_size 16 --image_size 64 --num_classes 10 --distributed True --main_gpu 0 --world_size 2
   ```

   **Unconditional Distributed Training Command**

   ```bash
   python train.py --conditional False --run_name 'df' --epochs 300 --batch_size 16 --image_size 64 --distributed True --main_gpu 0 --world_size 2
   ```

4. Wait for the training to complete. Interrupt recovery is the same as basic training.

**Parameter Explanation**

| Parameter Name         | Conditional | Usage                           | Type | Description                                                  |
| ---------------------- | :---------: | ------------------------------- | :--: | ------------------------------------------------------------ |
| --conditional          |             | Enable conditional training     | bool | Enable to modify custom configurations, such as modifying the number of classes and classifier-free guidance interpolation weights |
| --run_name             |             | File name                       | str  | File name used to initialize the model and save information  |
| --epochs               |             | Total number of epochs          | int  | Total number of training epochs                              |
| --batch_size           |             | Training batch size             | int  | Size of each training batch                                  |
| --num_workers          |             | Number of loading processes     | int  | Number of subprocesses used for data loading. It consumes a large amount of CPU and memory but can speed up training |
| --image_size           |             | Input image size                | int  | Input image size. Adaptive input and output sizes            |
| --dataset_path         |             | Dataset path                    | str  | Path to the conditional dataset, such as CIFAR-10, with each class in a separate folder, or the path to the unconditional dataset with all images in one folder |
| --fp16                 |             | Half precision training         | bool | Enable half precision training. It effectively reduces GPU memory usage but may affect training accuracy and results |
| --optim                |             | Optimizer                       | str  | Optimizer selection. Currently supports Adam and AdamW       |
| --lr                   |             | Learning rate                   | int  | Initial learning rate. Currently only supports linear learning rate |
| --result_path          |             | Save path                       | str  | Path to save the training results                            |
| --save_model_interval  |             | Save model after each training  | bool | Whether to save the model after each training iteration for model selection based on visualization |
| --start_model_interval |             | Start epoch for saving models   | int  | Start epoch for saving models. This option saves disk space. If not set, the default is -1. If set, it starts saving models from the specified epoch. It needs to be used with --save_model_interval |
| --vis                  |             | Visualize dataset information   | bool | Enable visualization of dataset information for model selection based on visualization |
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
   python generate.py --conditional True --generate_name 'df' --num_images 8 --num_classes 10 --class_name 0 --image_size 64 --weight_path '/your/path/weight/model.pt'
   ```

   **Unconditional Generation Command**

   ```bash
   python generate.py --conditional False --generate_name 'df' --num_images 8 --image_size 64 --weight_path '/your/path/weight/model.pt'
   ```

3. Wait for the generation process to complete.

**Parameter Explanation**

| Parameter Name  | Conditional | Usage                           | Type | Description                                                  |
| --------------- | :---------: | ------------------------------- | :--: | ------------------------------------------------------------ |
| --conditional   |             | Enable conditional generation   | bool | If enabled, allows custom configuration, such as modifying classes or classifier-free guidance interpolation weights |
| --generate_name |             | File name                       | str  | File name to initialize the model for saving purposes        |
| --image_size    |             | Input image size                | int  | Size of input images, adaptive input/output size             |
| --num_images    |             | Number of generated images      | int  | Number of images to generate                                 |
| --weight_path   |             | Path to model weights           | str  | Path to the model weights file, required for network generation |
| --result_path   |             | Save path                       | str  | Path to save the generated images                            |
| --num_classes   |      ✓      | Number of classes               | int  | Number of classes for classification                         |
| --class_name    |      ✓      | Class name                      | int  | Index of the class to generate images for                    |
| --cfg_scale     |      ✓      | Classifier-free guidance weight | int  | Weight for classifier-free guidance interpolation, for better generation model performance |

### Deployment

To be continued.

### Acknowledgements

[@dome272](https://github.com/dome272/Diffusion-Models-pytorch)

[@OpenAi](https://github.com/openai/improved-diffusion)