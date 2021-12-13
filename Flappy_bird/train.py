import os
import torch
from torch.autograd import Variable
import torch.nn as nn
import cv2
import sys
sys.path.append("game/")

import random
import numpy as np
from collections import deque

from create_game import GameState
from config import cfg
from net import Net

use_cuda = torch.cuda.is_available()
net = Net()
net.init()
net = net.cuda() if use_cuda else net

criterion = nn.MSELoss().cuda()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-6)

# 開啟遊戲程序
gameState = GameState()

# Replay Memory，類似 list 的儲存容器
D = deque()

# 將遊戲設定為初始狀態，並獲得一個 80x80 的遊戲畫面
do_nothing = np.zeros(cfg._ACTIONS_NUM)
do_nothing[0] = 1
x_t, r_0, terminal = gameState.frame_step(do_nothing)
x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)
ret, x_t = cv2.threshold(x_t, 1, 255, cv2.THRESH_BINARY)

# 將初始的遊戲畫面覆蓋成 4 張，作為初始 s_t
s_t = np.stack((x_t, x_t, x_t, x_t), axis=0)

# 設定初始的 epsilon，並準備 trainingepsilon = INITIAL_EPSILON
epsilon = cfg._INITIAL_EPSILON
t = 0

# 設定初始的 epsilon，並準備 trainingepsilon = INITIAL_EPSILON
# 紀錄每輪平均得分的容器
scores = []
all_turn_scores = []

# start game iter
while 'flappy bird' != 'angry bird':
    # 首先，按照 greedy 選擇一個動作
    s = Variable(torch.from_numpy(s_t).type(torch.FloatTensor))
    s = s.cuda() if use_cuda else s
    s = s.view(-1, s.size()[0], s.size()[1], s.size()[2])

    # 取得目前時刻的遊戲畫面，輸入神經網路中
    readout, h_fc1 = net(s)
    
    # 神經網路產生的輸出為 readout: 選擇每一個行動產生的預期 Q 值
    readout = readout.cpu() if use_cuda else readout
    readout_t = readout.data.numpy()[0]

    # 按照 epsilon greedy 產生小鳥的行動，即以 epsilon 的機率隨機行動
    # 或以 1-epsilon 的機率按照預期輸出最大的 Q 值列出動作
    a_t = np.zeros([cfg._ACTIONS_NUM])
    action_index = 0
    if t % cfg._FRAME_PER_ACTION == 0:
        if random.random() <= epsilon:
            # create random action
            # print("----------------Random Action----------------")
            action_index = random.randrange(cfg._ACTIONS_NUM)
        else:
            action_index = np.argmax(readout_t)

        a_t[action_index] = 1
    else:
        a_t[0] = 1  # do-nothing

    # 降低 epsilon
    if epsilon > cfg._FINAL_EPSILON and t > cfg._OBSERVE:
        epsilon -= (cfg._INITIAL_EPSILON - cfg._FINAL_EPSILON) / cfg._EXPLORE

    # 其次，將選擇好的行動輸入給遊戲引擎，並獲得下一幀狀態
    x_t1_colored, r_t, terminal = gameState.frame_step(a_t)

    # record score in every step
    scores.append(r_t)
    if terminal:
        # 當遊戲結束時，計算本輪的總成績，並將總成績輸入到 all_turn_scores
        all_turn_scores.append(sum(scores))
        scores = []

    # 對遊戲畫面進行處理，進一步變成一張 80x80 的無背景圖
    x_t1 = cv2.cvtColor(cv2.resize(x_t1_colored, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, x_t1 = cv2.threshold(x_t1, 1, 255, cv2.THRESH_BINARY)
    x_t1 = np.reshape(x_t1, (1, 80, 80))
    s_t1 = np.append(x_t1, s_t[:3, :, :], axis=0)  # (0~2, 80, 80)

    # 產生一個訓練資料對，並將其存入 D
    D.append((s_t, a_t, r_t, s_t1, terminal))
    if len(D) > cfg._REPLAY_MEMORY:
        D.popleft()

    if t > cfg._OBSERVE:
        # 從 D 中隨機取出一筆資料做訓練
        mini_batch = random.sample(D, cfg._BATCH_SIZE)
        optimizer.zero_grad()

        # 將這個 batch 中的變數全部分別存到清單中
        s_j_batch = [d[0] for d in mini_batch]
        a_batch = [d[1] for d in mini_batch]
        r_batch = [d[2] for d in mini_batch]
        s_j1_batch = [d[3] for d in mini_batch]

        # 接下來要經過 s_j1_batch 預測未來的 Q 值
        s = Variable(torch.FloatTensor(np.array(s_j1_batch, dtype=float)))
        s = s.cuda() if use_cuda else s
        readout, h_fc1 = net(s)
        readout = readout.cpu() if use_cuda else readout
        readout_j1_batch = readout.data.numpy()

        # 根據 Q 的預測值、目前回饋 r 與 terminal，更新待訓練的目標函數
        y_batch = []
        for i in range(len(mini_batch)):
            terminal = mini_batch[i][4]

            # 當遊戲結束的時候，則用環境的回饋作為目標
            # 否則用下一狀態的 Q 值加本期的環境回饋
            if terminal:
                y_batch.append(r_batch[i])
            else:
                # readout_j1_batch[i] 儲存了 32 個 [不扇, 扇]
                # np.max 會不分 dimesion 取最大數值
                y_batch.append(r_batch[i] +  cfg._GAMMA * np.max(readout_j1_batch[i]))
                

        # 開始梯度更新
        y = Variable(torch.FloatTensor(y_batch))
        a = Variable(torch.FloatTensor(a_batch))
        s = Variable(torch.FloatTensor(np.array(s_j_batch, dtype=float)))
        if use_cuda:
            y = y.cuda()
            a = a.cuda()
            s = s.cuda()

        # 計算 s_j_batch 的 Q 值
        readout, h_fc1 = net(s)
        # 被選擇的 action 會為 1，另一動作為 0，因此相乘後會留下被選擇的動作的 Q 值
        readout_action = readout.mul(a).sum(1)
        # readout_action 為依照舊有的 action 計算的預估 Q 值
        # y 則為被 Q(s',a') 影響計算過的目標 Q 值
        loss = criterion(readout_action, y)
        loss.backward()
        optimizer.step()
        if t % 1000 == 0:
            print('loss:', loss)

    # 將狀態更新一次
    s_t = s_t1
    t += 1

    if not os.path.exists('saving_nets'):
        os.mkdir('saving_nets')

    # 每隔 10000 次循環，儲存網路
    if t % 10000 == 0:
        torch.save(net, 'saving_nets/' + cfg._GAME_NAME + '-dqn' + str(t) + '.txt')

    if t <= cfg._OBSERVE:
        state = 'observe'
    elif t > cfg._OBSERVE and t <= cfg._OBSERVE + cfg._EXPLORE:
        state = 'explore'
    else:
        state = 'train'

    if t % 1000 == 0:
        record = "time_step:{};\tstate:{};\tEpsilon:{};\tAction:{};\tReward:{};\tQ_MAX:{:e};\tAll_turn_scors:{};".format(
                t,
                state, 
                epsilon,
                action_index,
                r_t, 
                np.max(readout_t),
                np.mean(all_turn_scores[-1000:])
                )

        print(record)
        with open('log_file.txt', 'a') as f:
            f.write(record +'\n')

