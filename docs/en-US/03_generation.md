### Generation

#### Start Your First Generation

##### Command Generation

1. Open the `generate.py` file and locate the `--weight_path` parameter. Modify the path in the parameter to the path of your model weights, for example `/your/path/weight/model.pt`.

2. Set the necessary parameters such as `--conditional`,`--latent`, `--generate_name`, `--num_images`, `--class_name`, `--image_size`, `--result_path`, etc. If no parameters are set, the default settings will be used. There are two ways to set the parameters: one is to directly modify the `parser` in the `if __name__ == "__main__":` section of the `generate.py` file, and the other is to use the following commands in the console while in the `/your/path/Defect-Diffusion-Model/iddm/tools` directory:

   **Conditional Generation Command**

   ```bash
   python generate.py --generate_name df --num_images 8 --mode class --class_name 0 --image_size 64 --weight_path /your/path/weight/model.pt --sample ddpm
   ```

   **Unconditional Generation Command**

   ```bash
   python generate.py --generate_name df --num_images 8 --image_size 64 --weight_path /your/path/weight/model.pt --sample ddpm
   ```

   **Conditional Latent Generation Command**

   ```bash
   python generate.py --conditional --latent --generate_name df --num_images 8 --mode class --class_name 0 --image_size 64 --weight_path /your/path/weight/model.pt --sample ddim --autoencoder_ckpt /your/path/weight/autoencoder_model.pt
   ```

   **Unconditional Latent Generation Command**

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
setattr(args, "latent", True)
setattr(args, "autoencoder_ckpt", "/your/VAE/model/path/model.pt")
setattr(args, "weight_path", "/your/model/path/model.pt")
setattr(args, "result_path", "/your/save/path/")
setattr(args, "mode", "class")
setattr(args, "class_name", 0)
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

| Parameter Name     | Conditional | Usage                           |   Type   | Description                                                                                                                                                                                                                         |
|--------------------|:-----------:|---------------------------------|:--------:|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| --generate_name    |             | File name                       |   str    | File name to initialize the model for saving purposes.                                                                                                                                                                              |
| --latent           |             | Enable latent diffusion model   |   bool   | If enabled, the model will use latent diffusion.                                                                                                                                                                                    |
| --image_format     |             | Generated image format          |   str    | Generated image format, jpg/png/jpeg. Recommend to use png for better generation quality.                                                                                                                                           |
| --num_images       |             | Number of generated images      |   int    | Number of images to generate.                                                                                                                                                                                                       |
| --use_ema          |             | Set the EMA model               |   bool   | Set the EMA model for generation                                                                                                                                                                                                    |
| --weight_path      |             | Path to model weights           |   str    | Path to the model weights file, required for network generation.                                                                                                                                                                    |
| --autoencoder_ckpt |             | VAE model weight path           |   str    | VAE model weight path.                                                                                                                                                                                                              |
| --result_path      |             | Save path                       |   str    | Path to save the generated images.                                                                                                                                                                                                  |
| --sample           |             | Sampling method                 |   str    | Set the sampling method type, currently supporting DDPM and DDIM. **(No need to set for models after version 1.1.1)**                                                                                                               |
| --image_size       |             | Input image size                |   int    | Size of input images, adaptive input/output size. if class name is -1 and conditional is True, the model would output one image per class.                                                                                          |
| --use_gpu          |             | Set the use GPU                 |   int    | Set the use GPU in generate, input is GPU's id.                                                                                                                                                                                     |
| --mode             |      ✓      | Guided generation               |   str    | Select the generation mode, Category-guided or Text-guided generation.                                                                                                                                                              |
| --class_name       |      ✓      | Class name                      |   str    | Index of the class to generate images. if class name is `-1`, the model would output one image per class.                                                                                                                           |
| --text             |      ✓      | Text description                |   str    | TODO: Text description for generation.                                                                                                                                                                                              |
| --cfg_scale        |      ✓      | Classifier-free guidance weight |   int    | Weight for classifier-free guidance interpolation, for better generation model performance.                                                                                                                                         |
| --conditional      |             | Enable conditional generation   |   bool   | If enabled, allows custom configuration, such as modifying classes or classifier-free guidance interpolation weights.                                                                                                               |
| --network          |             | Training network                |   str    | Set the training network, currently supporting UNet, CSPDarkUNet. **(No need to set for models after version 1.1.1)**                                                                                                               |
| --act              |             | Activation function             |   str    | Activation function selection. Currently supports gelu, silu, relu, relu6 and lrelu. If you do not set the same activation function as the model, mosaic phenomenon will occur. **(No need to set for models after version 1.1.1)** |
| --num_classes      |      ✓      | Number of classes               |   int    | Number of classes for classification. **(No need to set for models after version 1.1.1)**                                                                                                                                           |

