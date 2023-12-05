#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Date   : 2023/12/5 10:18
    @Author : chairc
    @Site   : https://github.com/chairc
"""


class EMA:
    """
    Exponential Moving Average
    """

    def __init__(self, beta):
        """
        Initialize EMA
        :param beta: β
        """
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ema_model, current_model):
        """
        Update model average
        :param ema_model: EMA model
        :param current_model: Current model
        :return: None
        """
        for current_params, ema_params in zip(current_model.parameters(), ema_model.parameters()):
            old_weight, up_weight = ema_params, current_params.data
            ema_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old_weight, new_weight):
        """
        Update average
        :param old_weight: Old weight
        :param new_weight: New weight
        :return: new_weight or old_weight * self.beta + (1 - self.beta) * new_weight
        """
        if old_weight is None:
            return new_weight
        return old_weight * self.beta + (1 - self.beta) * new_weight

    def step_ema(self, ema_model, model, step_start_ema=2000):
        """
        EMA step
        :param ema_model: EMA model
        :param model: Original model
        :param step_start_ema: Start EMA step
        :return: None
        """
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    @staticmethod
    def reset_parameters(ema_model, model):
        """
        Reset parameters
        :param ema_model: EMA model
        :param model: Original model
        :return: None
        """
        ema_model.load_state_dict(model.state_dict())
