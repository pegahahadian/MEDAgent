import torch
import torch.nn as nn
import torch.nn.functional as F

class ImageNet(nn.Module):
    def __init__(self):
        super(ImageNet, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)

        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers (Updated input size)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # ✅ Fix: Input matches 7x7 feature maps
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # Conv1 -> ReLU -> Pool
        x = self.pool(F.relu(self.conv2(x)))  # Conv2 -> ReLU -> Pool

        x = torch.flatten(x, 1)  # Flatten the feature maps
        x = F.relu(self.fc1(x))  # Fully connected layer 1
        x = F.relu(self.fc2(x))  # Fully connected layer 2
        x = self.fc3(x)          # Output layer
        
        x = torch.sigmoid(x)  # Use sigmoid for binary classification
        return x

# Example: Create an instance of the model and print its architecture
model = ImageNet()
print(model)
torch.save(model.state_dict(), "knee_oa_model.pth")
