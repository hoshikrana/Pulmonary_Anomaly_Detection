import torch
import torch.nn as nn
import config


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, is_final: bool = False):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            )
        ]
        if not is_final:
            layers += [
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=config.DROPOUT_RATE),
            ]
        else:
            layers.append(nn.Tanh())
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Decoder(nn.Module):
    """
    latent_dim -> 512x512 image
    """
    def __init__(self, latent_dim: int = config.LATENT_DIM):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 256 * 8 * 8),
            nn.ReLU(inplace=True),
        )
        self.deconv_layers = nn.Sequential(
            DecoderBlock(256, 256),   # 8 -> 16
            DecoderBlock(256, 128),   # 16 -> 32
            DecoderBlock(128, 64),    # 32 -> 64
            DecoderBlock(64, 32),     # 64 -> 128
            DecoderBlock(32, 16),     # 128 -> 256
            DecoderBlock(16, 1, is_final=True),  # 256 -> 512
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z)
        x = x.view(x.size(0), 256, 8, 8)
        return self.deconv_layers(x)