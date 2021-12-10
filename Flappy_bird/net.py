import torch
import torch.nn as nn
import torch.nn.functional as F
from config import cfg

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # input size = 80x80, input  channel = 4, 因為要 4 幀的連續畫面網路會比較好收斂
        self.conv1 = nn.Conv2d(4, 32, 8, 4, padding = 2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 4, 2, padding = 1)
        self.conv3 = nn.Conv2d(64, 64, 3, 1, padding = 1)

        self.fc_sz = 1600
        # 書上網路架構圖是 1600 -> 512 -> ACTIONS，但程式碼打 256(original:512)
        self.fc1 = nn.Linear(self.fc_sz, 256)
        self.fc2 = nn.Linear(256, cfg._ACTIONS_NUM)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)

        x = F.relu(self.conv2(x))

        x = F.relu(self.conv3(x))

        x = x.view(-1, self.fc_sz) 
        x = F.relu(self.fc1(x))
        readout = self.fc2(x)

        return readout, x

    def init(self):
        self.conv1.weight.data = torch.abs(0.01 * torch.randn(self.conv1.weight.size()))
        self.conv2.weight.data = torch.abs(0.01 * torch.randn(self.conv2.weight.size()))
        self.conv3.weight.data = torch.abs(0.01 * torch.randn(self.conv3.weight.size()))
        self.fc1.weight.data = torch.abs(0.01 * torch.randn(self.fc1.weight.size()))
        self.fc2.weight.data = torch.abs(0.01 * torch.randn(self.fc2.weight.size()))

        self.conv1.bias.data = torch.ones(self.conv1.bias.size()) * 0.01
        self.conv2.bias.data = torch.ones(self.conv2.bias.size()) * 0.01
        self.conv3.bias.data = torch.ones(self.conv3.bias.size()) * 0.01
        self.fc1.bias.data = torch.ones(self.fc1.bias.size()) * 0.01
        self.fc2.bias.data = torch.ones(self.fc2.bias.size()) * 0.01

