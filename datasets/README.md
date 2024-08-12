### Data Training Format README

This document provides a guide on how to structure your dataset for training using the provided source code. The relevant source code for loading the dataset can be found in the `get_dataset` method within `utils/dataset.py`.

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

This method allows for easy and efficient loading and labeling of image datasets for training deep learning models. Make sure your dataset is organized correctly to take full advantage of this functionality.
