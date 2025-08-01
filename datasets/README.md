## Data Training Format README

This document provides a guide on how to structure your dataset for training using the provided source code. The relevant source code for loading the dataset can be found in the `get_dataset` method within `utils/dataset.py`.

### 1. Diffusion Model Dataset

#### Dataset Structure

Your dataset should be organized in a specific folder structure to facilitate automatic labeling using the `torchvision.datasets.ImageFolder` class. The structure is as follows:

```
dataset_dir
│
├── class_1
│   ├── image_1.jpg
│   ├── image_2.jpg
│   └── ...
│
├── class_2
│   ├── image_1.jpg
│   ├── image_2.jpg
│   └── ...
│
└── ...
```

- **`dataset_dir`**: This is the directory of your dataset.
- **`class_1`, `class_2`, ...**: These are the subdirectories representing different categories/classes in your dataset.
- **`image_1.jpg`, `image_2.jpg`, ...**: These are the image files within each category.

#### How It Works

The `torchvision.datasets.ImageFolder` class is designed to work with datasets structured as described above. It automatically:

- **Loads** the images from the specified dataset directory.
- **Assigns labels** based on the folder names (e.g., all images in `class_1/` will be labeled as `class_1`).

#### Usage

To load your dataset using this structure, you can use the `ImageFolder` class in your code as follows:

```python
from torchvision import datasets, transforms

# Define transformations (e.g., resizing, normalization, etc.)
transform = transforms.Compose([
    # transformations code here
])

# Load dataset
dataset = datasets.ImageFolder(root='dataset_path', transform=transform)
```

- **`dataset_path`**: Replace this with the actual path to your dataset.
- **`transform`**: You can specify any preprocessing steps, such as resizing or normalization, through this parameter.



### 2. Latent Diffusion Model Dataset

#### Dataset Structure

##### Autoencoder

Your dataset should be organized in a specific folder structure to facilitate automatic labeling using the `torchvision.datasets.ImageFolder` class. The structure is as follows:

```
dataset_dir
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

- **`dataset_dir`**：Replace this with the actual path to your dataset.
- **`train`, `val`, ...**：These are subdirectories representing the training and validation subdirectories in the dataset.
- **`images`**：表示数据集中存放图片的子目录。
- **`image_1.jpg`, `image_2.jpg`, ...**：These are the image files.

#####  Diffusion Model

> [!NOTE]
>
> For detailed dataset format, please refer to **1. Diffusion Model Dataset - Dataset Structure**


#### How It Works

> [!NOTE]
>
> For detailed dataset format, please refer to **1. Diffusion Model Dataset - How It Works**


#### Usage

> [!NOTE]
>
> For detailed dataset format, please refer to **1. Diffusion Model Dataset - Usage**



### 3. Super Resolution

#### Dataset Structure

为了便于使用超分辨率模型，你的数据集应该按照特定的文件夹结构进行组织。结构如下：

```
dataset_dir
│
├── train
│   ├── image_1.jpg
│   ├── image_2.jpg
│   └── ...
│
└── val
    ├── image_1.jpg
    ├── image_2.jpg
    └── ...
```

- **`dataset_dir`**：Replace this with the actual path to your dataset.
- **`train`, `val`, ...**：These are subdirectories representing the training and validation subdirectories in the dataset.
- **`image_1.jpg`, `image_2.jpg`, ...**：These are the image files within training and validation.

#### How It Works

The `iddm/sr/dataset.py SRDataset` class is designed to work with datasets structured as described above. It automatically:

- **Loads** the images from the specified dataset directory.

#### Usage

To load a dataset using this structure, you can use the `SRDataset` class in `iddm/sr/dataset.py` as follows in your code:

```python
import os
import torchvision
from PIL import Image
from torch.utils.data import Dataset
from iddm.config.setting import SR_MEAN, SR_STD

class SRDataset(Dataset):
    def __init__(self, image_size=64, dataset_path="", scale=4):
        self.image_size = image_size
        self.dataset_path = dataset_path
        self.image_datasets = os.listdir(self.dataset_path)
        self.scale = scale
        self.lr_transforms = torchvision.transforms.Compose([
            # Resize input size
            # torchvision.transforms.Resize(image_size)
            torchvision.transforms.Resize(size=(int(self.image_size), int(self.image_size))),
            # To Tensor Format
            torchvision.transforms.ToTensor(),
            # For standardization, the mean and standard deviation
            torchvision.transforms.Normalize(mean=SR_MEAN, std=SR_STD)
        ])
        self.hr_transforms = torchvision.transforms.Compose([
            # Resize input size
            # torchvision.transforms.Resize(image_size * scale)
            torchvision.transforms.Resize(size=(int(self.image_size * self.scale), int(self.image_size * self.scale))),
            # To Tensor Format
            torchvision.transforms.ToTensor(),
            # For standardization, the mean and standard deviation
            torchvision.transforms.Normalize(mean=SR_MEAN, std=SR_STD)
        ])

    def __getitem__(self, index):
        """
        Get sr data
        :param index: Index
        :return: lr_images, hr_images
        """
        # Resize low resolution
        image_name = self.image_datasets[index]
        image_path = os.path.join(self.dataset_path, image_name)
        image = Image.open(fp=image_path)
        image = convert_3_channels(image)
        hr_image = image.copy()
        lr_image = image.copy()
        hr_image = self.hr_transforms(hr_image)
        lr_image = self.lr_transforms(lr_image)
        return lr_image, hr_image

    def __len__(self):
        return len(self.image_datasets)
```

- **`dataset_path`**：Replace this with the actual path to your dataset.
- **`__getitem__`**：You can use this method to get both low-resolution and high-resolution image sets.



This method allows for easy and efficient loading and labeling of image datasets for training deep learning models. Make sure your dataset is organized correctly to take full advantage of this functionality.
