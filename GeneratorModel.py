import torch
import torch.nn as nn
import torch.nn.functional as F


class GeneratorModel(nn.Module):
    def __init__(self):
        super(GeneratorModel, self).__init__()

        # define down blocks
        self.res_block1 = nn.Sequential(
            nn.Conv2d(6, 128, kernel_size=3, stride=1, padding="same"),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.res_block2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding="same"),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )


    def forward(self, x):
        h1 = self.res_block
