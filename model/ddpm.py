#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Date   : 2023/6/15 17:12
    @Author : chairc
    @Site   : https://github.com/chairc
"""
import torch
import logging
import coloredlogs
from tqdm import tqdm

logger = logging.getLogger(__name__)
coloredlogs.install(level="INFO")


class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=256, device="cpu"):
        """
        扩散模型ddpm复现
        论文：《Denoising Diffusion Probabilistic Models》
        链接：https://arxiv.org/abs/2006.11239
        :param noise_steps: 噪声步长
        :param beta_start: β开始值
        :param beta_end: β结束值
        :param img_size: 图像大小
        :param device: 设备类型
        """
        # 噪声步长
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device

        self.beta = self.prepare_noise_schedule().to(device)
        # 公式α = 1 - β
        self.alpha = 1. - self.beta
        # 这里做α累加和操作
        self.alpha_hat = torch.cumprod(input=self.alpha, dim=0)

    def prepare_noise_schedule(self):
        """
        准备噪声schedule，可以自定义，可使用openai的schedule
        torch.linspace为指定的区间内生成一维张量，其中的值均匀分布
        :return: schedule
        """
        return torch.linspace(start=self.beta_start, end=self.beta_end, steps=self.noise_steps)

    def noise_images(self, x, time):
        """
        给图片增加噪声
        :param x:
        :param time:
        :return:
        """
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[time])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[time])[:, None, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def sample_time_steps(self, n):
        """
        采样时间步长
        :param n:
        :return:
        """
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, n, labels=None, cfg_scale=None):
        """
        采样
        :param model: 模型
        :param n: 采样图片个数
        :param labels: 标签
        :param cfg_scale:
        :return:
        """
        logger.info(f"Sampling {n} new images....")
        model.eval()
        with torch.no_grad():
            # 输入格式为[n, 3, img_size, img_size]
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            # reversed(range(1, self.noise_steps)为反向迭代整数序列
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                # 时间步长，创建大小为n的张量
                t = (torch.ones(n) * i).long().to(self.device)
                # 这里判断网络是否有条件输入，例如多个类别输入
                if labels is None and cfg_scale is None:
                    # 图像与时间步长输入进模型中
                    predicted_noise = model(x, t)
                else:
                    predicted_noise = model(x, t, labels)
                    if cfg_scale > 0:
                        # 无条件预测噪声
                        unconditional_predicted_noise = model(x, t, None)
                        # torch.lerp根据给定的权重，在起始值和结束值之间进行线性插值，公式：input + weight * (end - input)
                        predicted_noise = torch.lerp(unconditional_predicted_noise, predicted_noise, cfg_scale)
                # 拓展为4维张量，根据时间步长t获取值
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                # 只需要步长大于1的噪声，详细参考论文P4页Algorithm2的第3行
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                # 在每一轮迭代中用x计算x的t - 1，详细参考论文P4页Algorithm2的第4行
                x = 1 / torch.sqrt(alpha) * (
                        x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(
                    beta) * noise
        model.train()
        # 将值恢复到0和1的范围
        x = (x.clamp(-1, 1) + 1) / 2
        # 乘255进入有效像素范围
        x = (x * 255).type(torch.uint8)
        return x
