#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Date   : 2025/9/12 14:26
    @Author : chairc
    @Site   : https://github.com/chairc
"""
import torch
import torch.nn as nn

from transformers import CLIPTextModel, CLIPTokenizer

from iddm.config.setting import EMB_CHANNEL, TEXT_FEAT_CHANNEL


class TextConditionAdapter(nn.Module):
    """
    Text Condition Adapter
    Encodes text into features compatible with time embeddings
    """

    def __init__(self, text_encoder_name="openai/clip-vit-base-patch32", text_feat_channel=TEXT_FEAT_CHANNEL,
                 emb_channel=EMB_CHANNEL, device="cuda"):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(text_encoder_name)
        self.text_encoder = CLIPTextModel.from_pretrained(text_encoder_name).to(device)
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        self.proj = nn.Sequential(
            nn.Linear(text_feat_channel, emb_channel),
            nn.SiLU(),
            nn.Linear(emb_channel, emb_channel)
        ).to(device)

    def forward(self, text_labels):
        """
        Forward
        :param text_labels: Input text label
        :return: Text embedding
        """
        inputs = self.tokenizer(text_labels, padding="max_length",
                                max_length=self.tokenizer.model_max_length,
                                truncation=True, return_tensors="pt").to(self.text_encoder.device)
        with torch.no_grad():
            text_embeds = self.text_encoder(**inputs).last_hidden_state[:, 0, :]
        return self.proj(text_embeds)


class ClassConditionAdapter(nn.Module):
    """
    Category Condition Adapter
    Encodes categories into features compatible with time embeddings
    """

    def __init__(self, num_classes, emb_channel=EMB_CHANNEL):
        super().__init__()
        self.label_emb = nn.Embedding(num_embeddings=num_classes, embedding_dim=emb_channel)

    def forward(self, class_labels):
        """
        Forward
        :param class_labels: Input class label
        :return: Class embedding
        """
        return self.label_emb(class_labels)
