### Generation

#### Start Your First Generation

##### Command Generation

1. Open the `generate.py` file and locate the `--weight_path` parameter. Modify the path in the parameter to the path of your model weights, for example `/your/path/weight/model.pt`.

2. Set the necessary parameters such as `--conditional`,`--latent`, `--generate_name`, `--num_images`, `--num_classes`(**No need to set for models after version 1.1.4**), `--class_name`, `--image_size`, `--result_path`, etc. If no parameters are set, the default settings will be used. There are two ways to set the parameters: one is to directly modify the `parser` in the `if __name__ == "__main__":` section of the `generate.py` file, and the other is to use the following commands in the console while in the `/your/path/Defect-Diffusion-Model/tools` directory:

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
   python generate.py --conditional --generate_name df --num_images 8 --class_name 0 --image_size 64 --weight_path /your/path/weight/model.pt --sample ddpm --network unet --act gelu 
   ```

   **Unconditional Generation Command (version <= 1.1.1)**

   ```bash
   python generate.py --generate_name df --num_images 8 --image_size 64 --weight_path /your/path/weight/model.pt --sample ddpm --network unet --act gelu 
   ```

   **Conditional Latent Generation Command (version > 1.2.0)**

   ```bash
   python generate.py --conditional --latent --generate_name df --num_images 8 --class_name 0 --image_size 64 --weight_path /your/path/weight/model.pt --sample ddim --autoencoder_ckpt /your/path/weight/autoencoder_model.pt
   ```

   **Unconditional Latent Generation Command (version > 1.2.0)**

   ```bash
   python generate.py --latent --generate_name df --num_images 8 --image_size 64 --weight_path /your/path/weight/model.pt --sample ddim --autoencoder_ckpt /your/path/weight/autoencoder_model.pt
   ```

3. Wait for the generation process to complete.

##### Python generation

```python
from iddm.tools.generate import Generator, init_generate_args

# Initialize generation arguments, or you can configure them by entering the init_generate_args method
args = init_generate_args()
# Customize your parameters
setattr(args, "weight_path", "/your/model/path/model.pt")
setattr(args, "result_path", "/your/save/path/")
# ...
# Or use args["parameter_name"] = "your setting"
gen_model = Generator(gen_args=args, deploy=False)
# Number of generations
num_images = 2
for i in range(num_images):
   gen_model.generate(index=i)
```



#### Generation Parameters

**Parameter Explanation**

> [!WARNING]
>
> `--sample`, `--network` and `--act` do not need to set for models after **version 1.1.1**
>
> `--num_classes` do not need to set for models after **version 1.1.4**

| Parameter Name     | Conditional | Usage                           | Type | Description                                                  |
| ------------------ | :---------: | ------------------------------- | :--: | ------------------------------------------------------------ |
| --conditional      |             | Enable conditional generation   | bool | If enabled, allows custom configuration, such as modifying classes or classifier-free guidance interpolation weights. |
| --generate_name    |             | File name                       | str  | File name to initialize the model for saving purposes.       |
| --latent           |             | Enable latent diffusion model   | bool | If enabled, the model will use latent diffusion.             |
| --image_size       |             | Input image size                | int  | Size of input images, adaptive input/output size. if class name is -1 and conditional is True, the model would output one image per class. |
| --image_format     |             | Generated image format          | str  | Generated image format, jpg/png/jpeg. Recommend to use png for better generation quality. |
| --num_images       |             | Number of generated images      | int  | Number of images to generate.                                |
| --weight_path      |             | Path to model weights           | str  | Path to the model weights file, required for network generation. |
| --autoencoder_ckpt |             | VAE model weight path           | str  | VAE model weight path.                                       |
| --result_path      |             | Save path                       | str  | Path to save the generated images.                           |
| --use_gpu          |             | Set the use GPU                 | int  | Set the use GPU in generate, input is GPU's id.              |
| --sample           |             | Sampling method                 | str  | Set the sampling method type, currently supporting DDPM and DDIM. **(No need to set for models after version 1.1.1)** |
| --network          |             | Training network                | str  | Set the training network, currently supporting UNet, CSPDarkUNet. **(No need to set for models after version 1.1.1)** |
| --act              |             | Activation function             | str  | Activation function selection. Currently supports gelu, silu, relu, relu6 and lrelu. If you do not set the same activation function as the model, mosaic phenomenon will occur. **(No need to set for models after version 1.1.1)** |
| --num_classes      |      ✓      | Number of classes               | int  | Number of classes for classification. **(No need to set for models after version 1.1.1)** |
| --class_name       |      ✓      | Class name                      | int  | Index of the class to generate images. if class name is `-1`, the model would output one image per class. |
| --cfg_scale        |      ✓      | Classifier-free guidance weight | int  | Weight for classifier-free guidance interpolation, for better generation model performance. |

