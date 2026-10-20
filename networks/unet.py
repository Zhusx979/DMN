import torch
import torch.nn as nn


class DoubelConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DoubelConv, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channel, out_channel, 3, padding=1),
            nn.ReLU()
        )

    def forward(self, input):
        return self.net(input) + input

class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.Doubel1 = DoubelConv(32, 32)
        self.MaxPool1 = nn.Conv2d(32, 64, 2, stride=2, padding=0)
        self.Doubel2 = DoubelConv(64, 64)
        self.MaxPool2 = nn.Conv2d(64, 128, 2, stride=2, padding=0)
        self.Doubel3 = DoubelConv(128, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2, padding=0)
        self.Doubel4 = DoubelConv(64 + 64, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2, padding=0)
        self.Doubel5 = DoubelConv(64 + 32, 96)
        self.classifier = nn.Conv2d(96, 5, 1)

    def forward(self, input):
        x1 = self.Doubel1(self.conv1(input))
        x2 = self.Doubel2(self.MaxPool1(x1))
        x3 = self.Doubel3(self.MaxPool2(x2))
        x4 = self.up1(x3)
        x5 = self.Doubel4(torch.cat([x4, x2], dim=1))
        x6 = self.up2(x5)
        x7 = self.Doubel5(torch.cat([x6, x1], dim=1))
        return self.classifier(x7)

if __name__ == '__main__':
    input = torch.rand((1,1,256, 256))
    model = Unet()
    output = model(input)
    print(f"Input shape {input.shape}")
    print(f"output shape {output.shape}")