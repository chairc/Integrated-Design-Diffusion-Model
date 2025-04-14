### Results

We conducted training on the following 5 datasets using the `DDPM` sampler with an image size of `64*64`. we also enabled `conditional`, using the `gelu` activation function, `linear` learning function and  setting learning rate to `3e-4`. The datasets are `cifar10`, `NEUDET`, `NRSD-MN`, `WOOD` and `Animate face` in `300` epochs.

The results are shown in the following as:



#### cifar10 dataset

![cifar_244_ema](../../assets/cifar_244_ema.jpg)![cifar_294_ema](../../assets/cifar_294_ema.jpg)



#### NEU-DET dataset

![neudet_290_ema](../../assets/neudet_290_ema.jpg)![neudet_270_ema](../../assets/neudet_270_ema.jpg)![neudet_276_ema](../../assets/neudet_276_ema.jpg)![neudet_265_ema](../../assets/neudet_265_ema.jpg)![neudet_240_ema](../../assets/neudet_240_ema.jpg)![neudet_244_ema](../../assets/neudet_244_ema.jpg)![neudet_245_ema](../../assets/neudet_245_ema.jpg)![neudet_298_ema](../../assets/neudet_298_ema.jpg)



#### NRSD dataset

![nrsd_180_ema](../../assets/nrsd_180_ema.jpg)![nrsd_188_ema](../../assets/nrsd_188_ema.jpg)![nrsd_194_ema](../../assets/nrsd_194_ema.jpg)![nrsd_203_ema](../../assets/nrsd_203_ema.jpg)![nrsd_210_ema](../../assets/nrsd_210_ema.jpg)![nrsd_217_ema](../../assets/nrsd_217_ema.jpg)![nrsd_218_ema](../../assets/nrsd_218_ema.jpg)![nrsd_248_ema](../../assets/nrsd_248_ema.jpg)![nrsd_276_ema](../../assets/nrsd_276_ema.jpg)![nrsd_285_ema](../../assets/nrsd_285_ema.jpg)![nrsd_295_ema](../../assets/nrsd_295_ema.jpg)![nrsd_298_ema](../../assets/nrsd_298_ema.jpg)



#### WOOD dataset

![wood_495](../../assets/wood_495.jpg)



#### Animate face dataset (~~JUST FOR FUN~~)

![model_428_ema](../../assets/animate_face_428_ema.jpg)![model_440_ema](../../assets/animate_face_440_ema.jpg)![model_488_ema](../../assets/animate_face_488_ema.jpg)![model_497_ema](../../assets/animate_face_497_ema.jpg)![model_499_ema](../../assets/animate_face_499_ema.jpg)![model_459_ema](../../assets/animate_face_459_ema.jpg)



#### Base on the 64×64 model to generate 160×160 (every size) images (Industrial surface defect generation only)

Of course, based on the 64×64 U-Net model, we generate 160×160 `NEU-DET` images in the `generate.py` file (single output, each image occupies 21GB of GPU memory). **Attention this [[issues]](https://github.com/chairc/Integrated-Design-Diffusion-Model/issues/9#issuecomment-1886422210)**! If it's an image with defect textures where the features are not clear, generating a large size directly might not have these issues, such as in NRSD or NEU datasets. However, if the image contains a background with specific distinctive features, you may need to use super-resolution or resizing to increase the size, for example, in Cifar10, CelebA-HQ, etc. **If you really need large-sized images, you can directly train with large pixel images if there is enough GPU memory.** Detailed images are as follows:

![model_499_ema](../../assets/neu160_0.jpg)![model_499_ema](../../assets/neu160_1.jpg)![model_499_ema](../../assets/neu160_2.jpg)![model_499_ema](../../assets/neu160_3.jpg)![model_499_ema](../../assets/neu160_4.jpg)![model_499_ema](../../assets/neu160_5.jpg)

