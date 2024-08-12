### 数据训练格式说明

本文件提供了如何根据提供的源代码对数据集进行结构化的指导。加载数据集的相关源代码可以在 `utils/dataset.py` 中的 `get_dataset` 方法中找到。

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

这种方法可以方便且高效地加载和标注用于训练深度学习模型的图像数据集。请确保正确组织你的数据集，以充分利用此功能。
