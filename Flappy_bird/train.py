import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import cv2
import sys
sys.path.append("game/")

import random
import numpy as np
from collections import deque

from create_game.GameState import GameState

GAME_NAME = 'bird'
ACTIONS_NUM = 2
GAMMA = 0.99
OBSERVE = 10000.  # 訓練之前的時間步，需先觀察 10000 幀
EXPLORE = 3000000.  # epslion 開始逐層變小
FINAL_EPSLION = 0.0001
INITIAL_EPSLION = 0.1
REPLAY_MEMORY = 50000  # 最多記憶多少幀的訓練資料
BATH_SIZE = 32
FRAME_PER_ACTION = 1  # 每間隔多少時間完成一次有效的動作輸出

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # input size = 80x80, input  channel = 4, 因為要 4 幀的連續畫面網路會比較好收斂
        self.conv1 = nn.Conv2d(4, 32, 8, 4, padding = 2)
        self.pool = nn.Maxpool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 4, 2, padding = 1)
        self.conv3 = nn.Conv2d(64, 64, 3, 1, padding = 1)

        self.fc_sz = 1600
        # 書上網路架構圖是 1600 -> 512 -> ACTIONS，但程式碼打 256(original:512)
        self.fc1 = nn.Linear(self.fc_sz, 256)
        self.fc2 = nn.Linear(256, ACTIONS_NUM)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(F)

        x = self.pool(x)

        x = self.conv2(x)
        x  = F.relu(x)

        x = self.conv3(x)
        x = F.relu(x)

        x = self.fc1(x)
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


use_cuda = torch.cuda.is_available()
net = Net()
net.init()
net = net.cuda() if use_cuda else net

criterion = nn.MSELoss().cuda()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-6)

# 開啟遊戲程序
game_state =  GameState()

# Replay Memory，類似 list 的儲存容器
D = deque()

# 將遊戲設定為初始狀態，並獲得一個 80x80 的遊戲畫面
do_nothing = np.zeros(ACTIONS_NUM)
do_nothing[0] = 1
x_t, r_0, terminal = game_state.frame_step(do_nothing)
x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)
ret, x_t = cv2.threshold(x_t, 1, 255, cv2.THRESH_BINARY)

# 將初始的遊戲畫面覆蓋成 4 張，作為初始 s_t
s_t = np.stack((x_t, x_t, x_t, x_t), axis=0)

# 設定初始的 epsilon，並準備 training
epsilon = INITIAL_EPSLION
t = 0







