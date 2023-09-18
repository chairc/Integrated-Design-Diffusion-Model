### This is a data training format README

Detailed reference source code is in the `get dataset` method in `utilsutils.py`.

Automatically divide labels torchvision.datasets.ImageFolder(args.dataset_path, transform=transforms)

If the dataset is as follow:
dataset_path/class_1/image_1.jpg
dataset_path/class_1/image_2.jpg
...
dataset_path/class_2/image_1.jpg
dataset_path/class_2/image_2.jpg
...

'dataset_path' is the root directory of the dataset, 'class_1', 'class_2', etc. are different categories in the dataset, and each category contains several image files.

Use the 'ImageFolder' class to conveniently load image datasets with this folder structure, and automatically assign corresponding labels to each image.

You can specify the root directory where the dataset is located by passing the 'dataset_path' parameter, and perform operations such as image preprocessing and label conversion through other optional parameters.
