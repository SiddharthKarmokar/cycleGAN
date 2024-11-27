import torch
import torch.nn as nn


class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels,
                      out_channels,
                      kernel_size=4,
                      stride=stride,
                      padding=1,
                      bias=True,
                      padding_mode='reflect'),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.conv(x)


class Discriminator(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=64,
                      kernel_size=4,
                      stride=2,
                      padding=1,
                      padding_mode="reflect"
            ),
            nn.LeakyReLU(0.2)
        )

        self.process = nn.Sequential(
            conv_block(64, 128, 2),
            conv_block(128, 256, 2),
            conv_block(256, 512, 2),
            nn.Conv2d(in_channels=512,
                      out_channels=1,
                      kernel_size=4,
                      stride=1,
                      padding=1,
                      padding_mode="reflect"),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.initial(x)
        x = self.process(x)
        return x
