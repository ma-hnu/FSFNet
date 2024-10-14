import torch
import numpy as np
import torchvision
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.nn import functional as F

from models.FSFEncoder import *
from models.resnet50 import Backbone
from utils.load_weights import load_pretrained_weights


class CA_Module(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.linear1 = torch.nn.Linear(input_dim, input_dim)
        self.relu = nn.ReLU()
        self.linear2 = torch.nn.Linear(input_dim, input_dim)
        self.sigmod = nn.Sigmoid()

    def forward(self, x):
        x1 = self.linear1(x)
        x1 = self.relu(x1)
        x1 = self.linear2(x1)
        x1 = self.sigmod(x1)
        x = x * x1
        return x


class ClassificationHead(nn.Module):
    def __init__(self, input_dim: int, target_dim: int):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, target_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        y_hat = self.linear(x)
        return y_hat


class FSFNet(nn.Module):
    def __init__(self, img_size=224, num_classes=4, type="large"):
        super().__init__()
        if type == "small":
            depth = 4
        if type == "base":
            depth = 6
        if type == "large":
            depth = 8

        self.img_size = img_size
        self.num_classes = num_classes



        self.backbone = Backbone(50)
        ir_checkpoint = torch.load('pretrain/ir50.pth', map_location=lambda storage, loc: storage)
        self.backbone = load_pretrained_weights(self.backbone, ir_checkpoint)

        self.fsf_encoder = FSFEncoder(in_chans=196, embed_dim=256,
                                             depth=depth, num_heads=8, mlp_ratio=2.,
                                             drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, keep_rate=(0.6,))


        self.ca = CA_Module(input_dim=256)
        self.head = ClassificationHead(input_dim=256, target_dim=self.num_classes)

    def forward(self, x):
        x_ir = self.backbone(x)
        y_hat = self.fsf_encoder(x_ir)
        y_hat = self.ca(y_hat)
        y_feat = y_hat
        out = self.head(y_hat)
        return out, y_feat

