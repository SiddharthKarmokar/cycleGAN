import torch
import torch.nn as nn


class gen_conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, TYPE='down', activation=False, **kwargs):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      padding_mode="reflect",
                      **kwargs) if TYPE == 'down'
            else nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                **kwargs
            ),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True) if activation else nn.Identity()
        )

    def forward(self, x):
        return self.conv(x)


class res_block(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            gen_conv_block(channels, channels, kernel_size=3, padding=1),
            gen_conv_block(channels, channels, activation=False, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return x + self.block(x)

class Generator(nn.Module):
    def __init__(self, in_channels=3, num_residual_block=9):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=64,
                      kernel_size=7,
                      stride=1,
                      padding=3,
                      padding_mode="reflect"),
            nn.ReLU(inplace=True)
        )
        self.down = nn.Sequential(
            gen_conv_block(64, 64*2, TYPE='down', kernel_size=3, stride=2, padding=1),
            gen_conv_block(64*2, 64*4, TYPE='down', kernel_size=3, stride=2, padding=1)
        )
        self.residual = nn.Sequential(
            *[res_block(64*4) for _ in range(num_residual_block)]
        )

        self.up = nn.Sequential(
            gen_conv_block(64*4, 64*2, TYPE='up', kernel_size=3, stride=2, padding=1, output_padding=1),
            gen_conv_block(64*2, 64, TYPE='up', kernel_size=3, stride=2, padding=1, output_padding=1)
        )

        self.get_img = nn.Conv2d(in_channels=64, out_channels=in_channels, kernel_size=7, stride=1, padding=1, padding_mode="reflect")

    def forward(self, x):
        x = self.initial(x)
        x = self.down(x)
        x = self.residual(x)
        x = self.up(x)

        return self.get_img(x)

