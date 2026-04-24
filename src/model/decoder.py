import torch
import torch.nn as nn
import config

class DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, is_final=False):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch) if not is_final else None
        self.act = nn.ReLU(inplace=True) if not is_final else nn.Tanh()
        self.drop = nn.Dropout2d(config.DROPOUT_RATE) if not is_final else None
        self.fuse = nn.Conv2d(out_ch * 2, out_ch, kernel_size=1, bias=False) if not is_final else None

    def forward(self, x, skip=None):
        x = self.up(x)
        if self.bn is not None:
            x = self.bn(x)
            x = self.act(x)
            if self.drop is not None:
                x = self.drop(x)
        if skip is not None:
            # Resize skip if needed and concat
            if x.shape[2:] != skip.shape[2:]:
                skip = nn.functional.interpolate(skip, size=x.shape[2:], mode='bilinear')
            x = torch.cat([x, skip], dim=1)
            # Fuse with 1x1 conv
            x = self.fuse(x)
        if self.act is not None and isinstance(self.act, nn.Tanh):
            x = self.act(x)
        return x

class Decoder(nn.Module):
    def __init__(self, latent_dim=config.LATENT_DIM):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(latent_dim, 512*8*8), nn.ReLU())
        self.up1 = DecoderBlock(512, 512)
        self.up2 = DecoderBlock(512, 512)
        self.up3 = DecoderBlock(512, 256)
        self.up4 = DecoderBlock(256, 128)
        self.up5 = DecoderBlock(128, 64)
        self.final = DecoderBlock(64, 1, is_final=True)

    def forward(self, z, skips):
        x = self.fc(z).view(-1, 512, 8, 8)
        x = self.up1(x, skips[4]) # 8x8 -> 16x16, concat with s5 (16x16, 512ch)
        x = self.up2(x, skips[3]) # 16x16 -> 32x32, concat with s4 (32x32, 512ch)
        x = self.up3(x, skips[2]) # 32x32 -> 64x64, concat with s3 (64x64, 256ch)
        x = self.up4(x, skips[1]) # 64x64 -> 128x128, concat with s2 (128x128, 128ch)
        x = self.up5(x, skips[0]) # 128x128 -> 256x256, concat with s1 (256x256, 64ch)
        x = self.final(x)         # 256x256 -> 512x512
        return x