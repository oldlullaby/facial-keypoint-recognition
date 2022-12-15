import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as I

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 32, 5)        # 224 --> 224 - 5 + 1 = 220
        self.conv1_bn = nn.BatchNorm2d(32)      # 220 / 2 = 110     
        
        self.conv2 = nn.Conv2d(32, 64, 3)       # 110 --> 110 - 3 + 1 = 108
        self.conv2_bn = nn.BatchNorm2d(64)      # 108 / 2 = 54
        
        self.conv3 = nn.Conv2d(64, 128, 3)      # 54 --> 54 - 3 + 1 = 52
        self.conv3_bn = nn.BatchNorm2d(128)     # Pool: 52 / 2 = 26
        
        self.conv4 = nn.Conv2d(128, 256, 3)     # 26 --> 26 - 3 + 1 = 24
        self.conv4_bn = nn.BatchNorm2d(256)     # Pool: 24 / 2 = 12

        self.conv5 = nn.Conv2d(256, 512, 3)     # 12 --> 12 - 3 + 1 = 10
        self.conv5_bn = nn.BatchNorm2d(512)     # Pool: 10 / 2 = 5
       

        self.fc1 = nn.Linear(5 * 5 * 512, 1024) # 5 * 5 * 512
        self.fc1_bn = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 28)

        self.drop1 = nn.Dropout2d(p=0.1)
        self.drop2 = nn.Dropout2d(p=0.2)
        self.drop3 = nn.Dropout(p=0.45)
        self.pool = nn.MaxPool2d(2, 2)
        
        
    def forward(self, x):

        x = self.drop1(self.pool(F.relu(self.conv1_bn(self.conv1(x)))))
        x = self.drop2(self.pool(F.relu(self.conv2_bn(self.conv2(x)))))
        x = self.drop1(self.pool(F.relu(self.conv3_bn(self.conv3(x)))))
        x = self.drop1(self.pool(F.relu(self.conv4_bn(self.conv4(x)))))
        x = self.drop2(self.pool(F.relu(self.conv5_bn(self.conv5(x)))))

        x = x.view(x.size(0), -1)

        x = self.drop3(F.relu(self.fc1_bn(self.fc1(x))))
        x = self.fc2(x)
        
        return x
