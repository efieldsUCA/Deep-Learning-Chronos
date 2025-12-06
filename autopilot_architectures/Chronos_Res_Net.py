import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """
    A Residual Block with a Skip Connection.
    This enables 'Deep' learning without the vanishing gradient problem.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        # Main path
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut path (The Skip Connection)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        # Pass through main path
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # Add the original input (Residual connection)
        out += self.shortcut(x)
        
        # Final activation
        out = F.relu(out)
        return out

class ChronosResNet(nn.Module):
    def __init__(self):
        super(ChronosResNet, self).__init__()
        
        # --- INITIAL PROCESSING ---
        # Rapidly reduce image size to save FPS
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # --- RESIDUAL LAYERS (The "Brain") ---
        # Your teammate just stacks layers. You use smart blocks.
        self.layer1 = ResidualBlock(32, 32, stride=1)
        self.layer2 = ResidualBlock(32, 64, stride=2) 
        self.layer3 = ResidualBlock(64, 64, stride=1)
        
        # --- SAFETY NET ---
        # Adaptive pooling ensures output is ALWAYS 4x4x64, no matter the input size.
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
        
        # --- DECISION LAYERS ---
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128), # 1024 input features (Efficient!)
            nn.ReLU(),
            nn.Dropout(0.5),            # Prevents memorizing the track
            nn.Linear(128, 2)           # Output: Steering, Throttle
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = self.fc(x)
        return x

if __name__ == "__main__":
    from torchinfo import summary
    model = ChronosResNet()
    # Confirms it fits the Raspberry Pi memory constraints
    summary(model, input_size=(1, 3, 180, 200))
