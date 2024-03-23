
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

#TODO: Verify padding/stride #s
class Net(nn.Module):
    def __init__(self, num_classes=10):
        super(Net, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2) 

        self.conv1 = nn.LazyConv2d(96, kernel_size=11, stride=4, padding=1)
        self.conv2 = nn.LazyConv2d(256, kernel_size=5, padding=2)
        self.conv3 = nn.LazyConv2d(384, kernel_size=3, padding=1)
        self.conv4 = nn.LazyConv2d(384, kernel_size=3, padding=1)
        self.conv5 = nn.LazyConv2d(256, kernel_size=3, padding=1)

        self.fc1 = nn.LazyLinear(4096)
        self.fc2 = nn.LazyLinear(4096)
        self.fc3 = nn.LazyLinear(num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = F.relu(x)
        
        x = self.conv4(x)
        x = F.relu(x)        
        
        x = self.conv5(x)
        x = F.relu(x)
        x = self.pool(x)

        x = x.view(x.size(0), -1) #Flatten

        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.relu(x)

        x = self.fc3(x)
        return F.log_softmax(x, dim=1)