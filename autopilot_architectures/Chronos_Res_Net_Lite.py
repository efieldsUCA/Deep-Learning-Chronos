import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)

        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)

        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class ChronosResNetLite(nn.Module):
    """
    Optimized ResNet for embedded racing:
    - ~0.65M params
    - Fast inference on Raspberry Pi
    - Smooth gradients
    - Strong generalization for steering/throttle
    """
    def __init__(self):
        super().__init__()

        # ------- STEM -------
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # ------- RESNET BLOCK STACK -------
        self.layer1 = BasicBlock(32, 32, stride=1)
        self.layer2 = BasicBlock(32, 64, stride=2)
        self.layer3 = BasicBlock(64, 64, stride=1)
        self.layer4 = BasicBlock(64, 128, stride=2)
        self.layer5 = BasicBlock(128, 128, stride=1)

        # ------- ADAPTIVE POOL -------
        self.pool = nn.AdaptiveAvgPool2d((2, 2))  # stabilized small output

        # ------- FC HEAD -------
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 2 * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 2),  # steering, throttle
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.pool(x)
        x = self.fc(x)
        return x
