import torch.nn as nn

# 网络模型
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1, 6, 3, 1, 2), nn.ReLU(),
                                   nn.MaxPool2d(2, 2))

        self.conv2 = nn.Sequential(nn.Conv2d(6, 16, 5), nn.ReLU(),
                                   nn.MaxPool2d(2, 2))

        self.fc1 = nn.Sequential(nn.Linear(16 * 5 * 5, 120),
                                 nn.BatchNorm1d(120), nn.ReLU())

        self.fc2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.BatchNorm1d(84),
            nn.ReLU(),
            nn.Linear(84, 10))


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

