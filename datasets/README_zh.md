## 数据训练格式说明

本文件提供了如何根据提供的源代码对数据集进行结构化的指导。加载数据集的相关源代码可以在 `utils/dataset.py` 中的 `get_dataset` 方法中找到。

### 1. 扩散模型数据集格式

#### 数据集结构

为了便于使用 `torchvision.datasets.ImageFolder` 类自动标注，你的数据集应该按照特定的文件夹结构进行组织。结构如下：

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

- **`dataset_dir`**：这是存放你的数据集的目录。
- **`class_1`, `class_2`, ...**：这些是表示数据集中不同类别的子目录。
- **`image_1.jpg`, `image_2.jpg`, ...**：这些是每个类别中的图像文件。

#### 如何运作

`torchvision.datasets.ImageFolder` 类设计用于处理上述结构化的数据集。它会自动执行以下操作：

- **加载** 指定数据集目录中的图像。
- **分配标签** 基于文件夹名称（例如，所有 `class_1/` 文件夹中的图像都将被标记为 `class_1`）。

#### 使用方法

要使用这种结构加载数据集，你可以在代码中如下使用 `ImageFolder` 类：

```python
from torchvision import datasets, transforms

# 定义数据转换（例如，调整大小、归一化等）
transform = transforms.Compose([
    # 转换代码
])

# 加载数据集
dataset = datasets.ImageFolder(root='dataset_path', transform=transform)
```

- **`dataset_path`**：将其替换为你数据集的实际路径。
- **`transform`**：你可以通过此参数指定任何预处理步骤，例如调整大小或归一化。



### 2. 潜扩散模型数据集格式

#### 数据集结构

##### 变分自编码器

为了便于使用 `torchvision.datasets.ImageFolder` 类自动标注，你的变分自编码器数据集应该按照特定的文件夹结构进行组织。结构如下：

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

- **`dataset_dir`**：这是存放你的数据集的目录。
- **`train`, `val`, ...**：这些是表示数据集中训练和验证的子目录。
- **`images`**：表示数据集中存放图片的子目录。
- **`image_1.jpg`, `image_2.jpg`, ...**：这些是每个文件夹中的图像文件。

#####  扩散模型

> [!NOTE]
>
> 详细数据集格式请参考：**1. 扩散模型数据集格式 - 数据集结构**
>


#### 如何运作

> [!NOTE]
>
> 详细数据集格式请参考：**1. 扩散模型数据集格式 - 如何运作**
>


#### 使用方法

> [!NOTE]
>
> 详细数据集格式请参考：**1. 扩散模型数据集格式 - 使用方法**
>



### 3. 超分辨率数据集格式

#### 数据集结构

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

- **`dataset_dir`**：这是存放你的数据集的目录。
- **`train`, `val`, ...**：这些是表示数据集中训练和验证的子目录。
- **`image_1.jpg`, `image_2.jpg`, ...**：这些是训练和验证中的图像文件。

#### 如何运作

在`iddm/sr/dataset.py`的`SRDataset`类设计用于处理上述结构化的数据集。它会自动执行以下操作：

- **加载** 指定数据集目录中的图像。

#### 使用方法

要使用这种结构加载数据集，你可以在代码中如下使用 `iddm/sr/dataset.py`的`SRDataset`类：

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
            # 调整输入尺寸
            # torchvision.transforms.Resize(image_size)
            torchvision.transforms.Resize(size=(int(self.image_size), int(self.image_size))),
            # Tensor格式
            torchvision.transforms.ToTensor(),
            # 均值和标准化
            torchvision.transforms.Normalize(mean=SR_MEAN, std=SR_STD)
        ])
        self.hr_transforms = torchvision.transforms.Compose([
            # 调整输入尺寸
            # torchvision.transforms.Resize(image_size * scale)
            torchvision.transforms.Resize(size=(int(self.image_size * self.scale), int(self.image_size * self.scale))),
            # Tensor格式
            torchvision.transforms.ToTensor(),
            # 均值和标准化
            torchvision.transforms.Normalize(mean=SR_MEAN, std=SR_STD)
        ])

    def __getitem__(self, index):
        """
        获取超分辨率图像组
        :param index: Index
        :return: lr_images, hr_images
        """
        # 调整低分辨率尺寸
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

- **`dataset_path`**：将其替换为你数据集的实际路径。
- **`__getitem__`**：你可以通过此方法获取到低分辨率和高分辨率图像组。



这种方法可以方便且高效地加载和标注用于训练深度学习模型的图像数据集。请确保正确组织你的数据集，以充分利用此功能。
