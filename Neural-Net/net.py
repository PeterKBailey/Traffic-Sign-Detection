
import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self, num_classes=43):
        super(LeNet, self).__init__()
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2) 

        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)

        self.fc1 = nn.Linear(400, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        self.dropout = nn.Dropout(0.25)


    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)

        x = torch.flatten(x, 1) #Flatten

        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc3(x)
        return F.log_softmax(x, dim=1)
    
class VGNet(nn.Module):
    def __init__(self, num_classes=43, dropout=0):
        super(VGNet, self).__init__()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2) 

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)

        self.fc1 = nn.Linear(256*4*4, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, num_classes)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        #Block 1
        x = self.conv1(x)
        x = F.relu(x)
        x = self.max_pool(x)

        #Block 2
        x = self.conv2(x)
        x = F.relu(x)
        x = self.max_pool(x)

        #Block 3
        x = self.conv3(x)
        x = F.relu(x)
        x = self.max_pool(x)

        x = torch.flatten(x, 1) #Flatten

        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc3(x)
        return F.log_softmax(x, dim=1)
    
    