### Evaluation

#### Start Your First Evaluation

1. During the data preparation stage, use `generate.py` to create the dataset. The amount and size of the generated dataset should be similar to the training set.

   > [!TIP]
   >
   > The training set required for evaluation should be resized to the size used during training, which is the `image_size`.
   >
   > For example, if the training set path is `/your/path/datasets/landscape` with an image size of **256**, and the generated set path is `/your/path/generate/landscape` with a size of 64, use the `resize` method to convert the images in the training set path to **64**. The new evaluation training set path will be `/your/new/path/datasets/landscape`.

2. Open the `FID_calculator.py` or `FID_calculator_plus.py` file for evaluation. `FID_calculator.py` is for **simple evaluation**; `FID_calculator_plus.py` is for **custom evaluation**, allowing various parameter settings.

3. If using `FID_calculator.py`, set `generated_image_folder` to `/your/path/generate/landscape` and `dataset_image_folder` to `/your/new/path/datasets/landscape`. **Right-click to run**.

4. If using `FID_calculator_plus.py`, set the necessary parameters such as `path`, `--batch_size`, `--num-workers`, `--dims`, `--save_stats`, and `--use_gpu`. If no parameters are set, the default settings will be used. There are two methods for setting parameters. One is to directly set the `parser` in the `if __name__ == "__main__":` block of the `FID_calculator_plus.py` file. The other is to enter the following command in the console under the `/your/path/Defect-Diffiusion-Model/iddm/tools` directory:  

   **For evaluation only**

   ```bash
   python FID_calculator_plus.py /your/path/generate/landscape /your/new/path/datasets/landscape --batch_size 8 --num-workers 2 --dims 2048 --use_gpu 0
   ```

   **To generate npz archives** (**generally not needed**)

   ```bash
   python FID_calculator_plus.py /your/input/path /your/output/path --save_stats
   ```



#### Evaluation Parameters

| **Parameter Name** | Usage                       | Parameter Type | Explanation                                                  |
| ------------------ | --------------------------- | :------------: | ------------------------------------------------------------ |
| path               | Path                        |      str       | Input two paths: the generated set path and the training set path in evaluation mode; input path and output path in npz mode |
| --batch_size       | Training batch size         |      int       | Size of each training batch                                  |
| --num_workers      | Number of loading processes |      int       | Number of subprocesses used for data loading. It consumes a large amount of CPU and memory but can speed up training |
| --dims             | Dimensions                  |      int       | The dimensions of the Inception features to use              |
| --save_stats       | Save stats                  |      bool      | Generate npz archives from the sample directory              |
| --use_gpu          | Specify GPU                 |      int       | Generally used to set the specific GPU for training, input the GPU number |
