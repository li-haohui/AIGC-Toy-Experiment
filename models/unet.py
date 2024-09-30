import torch
import torch.nn as nn
import torch.nn.functional as F

from common.registry import registry

# 定义卷积块
class DoubleConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        activation=nn.ReLU(inplace=True)
    ):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            activation,
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            activation
        )

    def forward(self, x):
        return self.conv(x)

# 定义UNet模型
@registry.register_model("unet")
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        self.time_embedding = nn.Embedding(1000, embedding_dim=64)

        self.encoder1 = DoubleConv(in_channels, 64)
        self.encoder2 = DoubleConv(64, 128)
        self.encoder3 = DoubleConv(128, 256)
        self.encoder4 = DoubleConv(256, 512)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = DoubleConv(512, 1024)

        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder4 = DoubleConv(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = DoubleConv(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = DoubleConv(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = DoubleConv(128, 64)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x, time_step):

        t_emb = self.time_embedding(time_step)

        # 编码器路径
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool(e1))
        e3 = self.encoder3(self.pool(e2))
        e4 = self.encoder4(self.pool(e3))

        # 瓶颈层
        b = self.bottleneck(self.pool(e4))

        # 解码器路径
        d4 = self.upconv4(b)
        d4 = torch.cat((e4, d4), dim=1)  # 跳跃连接
        d4 = self.decoder4(d4)

        d3 = self.upconv3(d4)
        d3 = torch.cat((e3, d3), dim=1)
        d3 = self.decoder3(d3)

        d2 = self.upconv2(d3)
        d2 = torch.cat((e2, d2), dim=1)
        d2 = self.decoder2(d2)

        d1 = self.upconv1(d2)
        d1 = torch.cat((e1, d1), dim=1)
        d1 = self.decoder1(d1)

        # 输出层
        out = self.final_conv(d1)
        return out

    @classmethod
    def from_config(cls, cfg):
        in_channels = cfg.in_channels
        out_channels = cfg.out_channels

        return cls(in_channels, out_channels)

# 检查模型是否正常运行
if __name__ == "__main__":
    model = UNet(in_channels=3, out_channels=3)
    x = torch.randn(1, 3, 32, 32)  # 输入为 (batch_size, in_channels, height, width)
    y = model(x)
    print(y.shape)  # 输出维度应为 (batch_size, out_channels, height, width)
