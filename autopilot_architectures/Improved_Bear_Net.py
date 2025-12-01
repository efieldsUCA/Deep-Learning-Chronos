import torch
import torch.nn as nn

class ImprovedBearNet(nn.Module):
    def __init__(self):
        super(ImprovedBearNet, self).__init__()
        
        # Convolutional layers with Batch Normalization and smaller kernels
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.conv7 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm2d(256)

        # ReLU activation
        self.relu = nn.ReLU()

        # Max Pooling or Adaptive Pooling
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))  # Fixed size output

        # Fully connected layers with dropout for regularization
        self.fc1 = nn.Linear(256 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 2)  # Output layer for steering and throttle
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Convolutional layers with ReLU activations and batch normalization
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.max_pool(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.max_pool(x)
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.relu(self.bn5(self.conv5(x)))
        x = self.max_pool(x)
        x = self.relu(self.bn6(self.conv6(x)))
        x = self.relu(self.bn7(self.conv7(x)))
        
        # Adaptive pooling to a fixed 7x7 output
        x = self.adaptive_pool(x)

        # Flatten the tensor for fully connected layers
        x = x.view(x.size(0), -1)
        
        # Fully connected layers with dropout
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)  # Final output
        
        return x


if __name__ == "__main__":
    from torchinfo import summary

    model = ImprovedBearNet()  # Adjust num_classes as needed
    batch_size = 1
    summary(model, input_size=(batch_size, 3, 180, 200))  # Use the target image resolution (180x200)
