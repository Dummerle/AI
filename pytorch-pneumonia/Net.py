import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv3 = nn.Conv2d(20, 25, kernel_size=5)
        self.conv4= nn.Conv2d(25, 35, kernel_size=5)
        self.fc1 = nn.Linear(5915, 1000)
        self.fc2 = nn.Linear(1000, 3)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 3)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 3)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.max_pool2d(x, 3)
        x = F.relu(x)
        x=self.conv4(x)
        x=F.max_pool2d(x,3)
        x=F.relu(x)
        x=x.view(-1, 5915)
        x = F.relu(self.fc1(x))
        x= self.fc2(x)
        return F.softmax(x)

