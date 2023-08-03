#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Date   : 2023/7/21 23:06
    @Author : chairc
    @Site   : https://github.com/chairc
"""
import math

import torch


class BaseDiffusion:
    """
    扩散模型基类
    """
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=256, device="cpu"):
        """
        扩散模型基类
        :param noise_steps: 噪声步长
        :param beta_start: β开始值
        :param beta_end: β结束值
        :param img_size: 图像大小
        :param device: 设备类型
        """
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device

        # 噪声步长
        self.beta = self.prepare_noise_schedule().to(self.device)
        # 公式α = 1 - β
        self.alpha = 1. - self.beta
        # 这里做α累加和操作
        self.alpha_hat = torch.cumprod(input=self.alpha, dim=0)

    def prepare_noise_schedule(self, schedule_name="linear"):
        """
        准备噪声schedule，可以自定义，可使用openai的schedule
        :param schedule_name: 方法名称，linear线性方法；cosine余弦方法
        :return: schedule
        """
        if schedule_name == "linear":
            # torch.linspace为指定的区间内生成一维张量，其中的值均匀分布
            return torch.linspace(start=self.beta_start, end=self.beta_end, steps=self.noise_steps)
        elif schedule_name == "cosine":
            def alpha_hat(t):
                """
                其参数t从0到1，并生成(1 - β)到扩散过程的该部分的累积乘积
                原式â计算公式为：α_hat(t) = f(t) / f(0)
                原式f(t)计算公式为：f(t) = cos(((t / (T + s)) / (1 + s)) · (π / 2))²
                在此函数中s = 0.008且f(0) = 1
                所以仅返回f(t)即可
                :param t: 时间
                :return: t时alpha_hat的值
                """
                return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2

            # 要产生的beta的数量
            noise_steps = self.noise_steps
            # 使用的最大β值；使用小于1的值来防止出现奇点
            max_beta = 0.999
            # 创建一个分散给定alpha_hat(t)函数的β时间表，从t = [0,1]定义了（1 - β）的累积产物
            betas = []
            # 循环遍历
            for i in range(noise_steps):
                t1 = i / noise_steps
                t2 = (i + 1) / noise_steps
                # 计算β在t时刻的值，公式为：β(t) = min(1 - (α_hat(t) - α_hat(t-1)), 0.999)
                beta_t = min(1 - alpha_hat(t2) / alpha_hat(t1), max_beta)
                betas.append(beta_t)
            return torch.tensor(betas)
        else:
            raise NotImplementedError(f"Unknown beta schedule: {schedule_name}")

    def noise_images(self, x, time):
        """
        给图片增加噪声
        :param x: 输入图像信息
        :param time: 时间
        :return: sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, t时刻形状与x张量相同的张量
        """
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[time])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[time])[:, None, None, None]
        # 生成一个形状与x张量相同的张量，其中的元素是从标准正态分布（均值为0，方差为1）中随机抽样得到的
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def sample_time_steps(self, n):
        """
        采样时间步长
        :param n: 图像尺寸
        :return: 形状为(n,)的整数张量
        """
        # 生成一个具有指定形状(n,)的整数张量，其中每个元素都在low和high之间（包含 low，不包含 high）随机选择
        return torch.randint(low=1, high=self.noise_steps, size=(n,))
