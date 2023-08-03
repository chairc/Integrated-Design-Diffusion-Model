#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Date   : 2023/7/7 9:55
    @Author : chairc
    @Site   : https://github.com/chairc
"""
import torch
import logging
import coloredlogs

from tqdm import tqdm

from .base import BaseDiffusion

logger = logging.getLogger(__name__)
coloredlogs.install(level="INFO")


class Diffusion(BaseDiffusion):
    """
    DDIM扩散模型
    """

    def __init__(self, noise_steps=1000, sample_steps=20, beta_start=1e-4, beta_end=0.02, img_size=256, device="cpu"):
        """
        扩散模型ddim复现
        论文：《Denoising Diffusion Implicit Models》
        链接：https://arxiv.org/abs/2010.02502
        :param noise_steps: 噪声步长
        :param sample_steps: 采样步长
        :param beta_start: β开始值
        :param beta_end: β结束值
        :param img_size: 图像大小
        :param device: 设备类型
        """
        super().__init__(noise_steps, beta_start, beta_end, img_size, device)
        # 采样步长，用于跳步
        self.sample_steps = sample_steps

        self.eta = 0

        # 计算迭代步长，跳步操作
        self.time_step = torch.arange(0, self.noise_steps, (self.noise_steps // self.sample_steps)).long() + 1
        self.time_step = reversed(torch.cat((torch.tensor([0], dtype=torch.long), self.time_step)))
        self.time_step = list(zip(self.time_step[:-1], self.time_step[1:]))

    def sample(self, model, n, labels=None, cfg_scale=None):
        """
        采样
        :param model: 模型
        :param n: 采样图片个数
        :param labels: 标签
        :param cfg_scale: classifier-free guidance插值权重，用于提升生成质量，避免后验坍塌（posterior collapse）问题
                            参考论文：《Classifier-Free Diffusion Guidance》
        :return: 采样图片
        """
        logger.info(msg=f"DDIM Sampling {n} new images....")
        model.eval()
        with torch.no_grad():
            # 输入格式为[n, 3, img_size, img_size]
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            # i和i的前一个时刻
            for i, p_i in tqdm(self.time_step):
                # t时间步长，创建大小为n的张量
                t = (torch.ones(n) * i).long().to(self.device)
                # t的前一个时间步长
                p_t = (torch.ones(n) * p_i).long().to(self.device)
                # 拓展为4维张量，根据时间步长t获取值
                alpha_t = self.alpha_hat[t][:, None, None, None]
                alpha_prev = self.alpha_hat[p_t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                # 这里判断网络是否有条件输入，例如多个类别输入
                if labels is None and cfg_scale is None:
                    # 图像与时间步长输入进模型中
                    predicted_noise = model(x, t)
                else:
                    predicted_noise = model(x, t, labels)
                    # 用于提升生成，避免后验坍塌（posterior collapse）问题
                    if cfg_scale > 0:
                        # 无条件预测噪声
                        unconditional_predicted_noise = model(x, t, None)
                        # torch.lerp根据给定的权重，在起始值和结束值之间进行线性插值，公式：input + weight * (end - input)
                        predicted_noise = torch.lerp(unconditional_predicted_noise, predicted_noise, cfg_scale)
                # 核心计算公式
                x0_t = torch.clamp((x - (predicted_noise * torch.sqrt((1 - alpha_t)))) / torch.sqrt(alpha_t), -1, 1)
                c1 = self.eta * torch.sqrt((1 - alpha_t / alpha_prev) * (1 - alpha_prev) / (1 - alpha_t))
                c2 = torch.sqrt((1 - alpha_prev) - c1 ** 2)
                x = torch.sqrt(alpha_prev) * x0_t + c2 * predicted_noise + c1 * noise
        model.train()
        # 将值恢复到0和1的范围
        x = (x + 1) * 0.5
        # 乘255进入有效像素范围
        x = (x * 255).type(torch.uint8)
        return x
