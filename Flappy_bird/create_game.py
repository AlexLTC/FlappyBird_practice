import numpy as np
import sys
import random
import pygame
import pygame.surfarray as surfarray
from pygame.locals import *
from itertools import cycle

import resource

FPS = 30
SCREENWIDTH = 288
SCREENHEIGHT = 512

pygame.init()
FPSCLOCK = pygame.time.Clock()  # 定義程式時脈
SCREEN = pygame.display.set_mode((SCREENWIDTH, SCREENHEIGHT))  # 定義螢幕物件
pygame.display.set_caption('*Flappy Bird*')  # 設定視窗名稱


IMAGES, SOUNDS, HITMASKS = resource.load()  # 載入遊戲資源
PIPEGAPSIZE = 100  # 定義水管間的距離
BASE_Y = SCREENHEIGHT * 0.79  # 設定基地的高度

# 設定小鳥的屬性：高度、寬度
PLAYER_WIDTH = IMAGES['player'][0].get_width()
PLAYER_HEIGHT = IMAGES['player'][0].get_height()

# 設定水管的屬性：高度、寬度
PIPE_WIDTH = IMAGES['player'][0].get_width()
PIPE_HEIGHT = IMAGES['player'][0].get_height()

# 背景寬度設定
BACKGROUND_WIDTH = IMAGES['background'].get_width()

# example: cycle('ABCD') --> A B C D A B C D ...
PLAYER_INDEX_GEN = cycle([0, 1, 2, 1])

def getRandomPipe():
    # 兩個管線之間的垂直間隔從下列數中直接取
    gapYs = [20, 30, 40, 50, 60, 70, 80, 90]
    index = random.randint(0, len(gapYs)-1)
    gapY = gapYs[index]

    # 設定新產生管線的位置
    gapY += int(BASE_Y * 0.2)
    pipeX = SCREENWIDTH + 10

    # 傳回管線的座標
    return [
            {'x':pipeX, 'y':gapY - PIPE_HEIGHT}, # upper pipe
            {'x':pipeX, 'y':gapY + PIPEGAPSIZE}, # lower pipe
    ]


def pixelCollision(rect1, rect2, hitMask1, hitMask2):
    # 在像素等級檢查兩個物體是否發生碰撞
    rect = rect1.clip(rect2)
    if rect.width == 0 or rect.height == 0:  # two rectangles do not overlap
        return False

    # 確定矩形框，並針對舉行框中的每個像素進行循環，檢視兩個物件是否碰撞
    x1, y1 = rect.x - rect1.x, rect.y - rect1.y
    x2, y2 = rect.x - rect2.x, rect.y - rect2.y

    for x in range(rect.width):
        for y in range(rect.height):
            if hitMask1[x1+x][y1+y] and hitMask2[x2+x][y2+y]:
                return True

    return False

# 檢測物件碰撞，以遮色片等級偵測，並非單純看矩形之間的碰撞
def checkCrash(player, upperPipes, lowerPipes):
    player_index = player['index']
    player['w'] = IMAGES['player'][0].get_width()
    player['h'] = IMAGES['player'][0].get_height()
    
    # 檢查小鳥是否碰撞到地面
    if player['y'] + player['h'] >= BASE_Y - 1:
        return True
    else:
        # 檢查小鳥是否與 Pipe 碰撞
        playerRect = pygame.Rect(player['x'], player['y'], PIPE_WIDTH, PIPE_HEIGHT)  # make player's rectangle
        for uPipe, lPipe in zip(upperPipes, lowerPipes):
            uPipeRect = pygame.Rect(uPipe['x'], uPipe['y'], PIPE_WIDTH, PIPE_HEIGHT)  # make upper Pipe's rectangle
            lPipeRect = pygame.Rect(lPipe['x'], lPipe['y'], PIPE_WIDTH, PIPE_HEIGHT)  # make lower Pipes's rectangle
        
        # 獲得每個元素的遮色片
        pHitMask = HITMASKS['player'][player_index]
        uHitMask = HITMASKS['pipe'][0]
        lHitMask = HITMASKS['pipe'][1]

        # 檢查是否與上下管線相撞
        uCollide = pixelCollision(playerRect, uPipeRect, pHitMask, uHitMask)
        lCollide = pixelCollision(playerRect, lPipeRect, pHitMask, lHitMask)
        if uCollide or lCollide:
            return True

    return False


class GameState:
    def __init__(self):
        # initialize
        # intialize score, player index, 反覆循環運算 equals to zero
        self.score = self.playerIndex = self.loopIter = 0

        # setting player's initial position
        self.playerX = int(SCREENWIDTH * 0.2)
        self.playerY = int((SCREENHEIGHT - PLAYER_HEIGHT) / 2)
        self.baseX = 0
        self.baseShift = IMAGES['base'].get_width() - BACKGROUND_WIDTH  # 地面的初始位移

        # 產生兩個隨機的水管
        newPipe1 = getRandomPipe()
        newPipe2 = getRandomPipe()

        #  設定初始水管的位置 (x, y) 座標
        self.upperPipes = [
                {'x': SCREENWIDTH, 'y': newPipe1[0]['y']},
                {'x': SCREENWIDTH + (SCREENWIDTH / 2), 'y': newPipe2[0]['y']}
        ]
        self.lowerPipes = [
                {'x': SCREENWIDTH, 'y': newPipe1[1]['y']},
                {'x': SCREENWIDTH + (SCREENWIDTH / 2), 'y': newPipe2[1]['y']}
        ]

        # 定義玩家的屬性
        self.pipeVelX = -4
        self.playerVelY = 0         # 小鳥在 y 軸上的速度
        self.playerMaxVelY = 10     # y 軸上的最大速度
        self.playerMinVelY = -8     # y 軸上的最小速度
        self.playerAccY = 1         # 小鳥下落的加速度
        self.playerFlapAcc = -9     # 扇動翅膀的加速度
        self.playerFlapped = False  # 玩家是否扇動了翅膀

    
    def frame_step(self, input_actions):
        # input_actions 是一個行動陣列，分別儲存了 0 或 1 兩個動作的啟動情況
        # 遊戲每一幀的循環
        pygame.event.pump()

        # 每一步的預設回報
        reward = 0.1
        terminal = False

        # 限定每一幀只能做一個動作
        if sum(input_actions) != 1:
            raise ValueError('Mutiple input actions detected!')

        # input_actions[0] == 1 : 對應什麼都沒做
        # input_actions[1] == 1 : 對應小鳥扇動了翅膀
        if input_actions[1] == 1:
            # 小鳥扇動翅膀在上
            if self.playerY > -2 * PLAYER_HEIGHT:
                self.playerVelY = self.playerFlapAcc
                self.playerFlapped = True

        # 檢查是否通過了管線，通過則給予獎勵
        playerMidPos = self.playerX + PLAYER_WIDTH / 2
        for pipe in self.upperPipes:
            pipeMidPos = pipe['x'] + PIPE_WIDTH / 2
            if pipeMidPos <= playerMidPos < pipeMidPos + 4:  # pipVelX = -4
                self.score += 1
                reward = 1

        # playerIndex 輪換
        if (self.loopIter + 1) % 3 == 0:
            self.player = next(PLAYER_INDEX_GEN)  # self.player 將會在 iter 中順序性的提出 PLAYER_INDEX_GEN 的值
            
        self.loopIter = (self.loopIter + 1) % 30
        self.baseX = -((-self.baseX + 100) % self.baseShift)  # 不懂

        # 小鳥運動
        if self.playerVelY < self.playerMaxVelY and not self.playerFlapped:
            self.playerVelY += self.playerAccY

        if self.playerFlapped:
            self.playerFlapped = False

        self.playerY += min(self.playerVelY, BASE_Y - self.playerY - PLAYER_HEIGHT)  # 不懂

        if self.playerY < 0:
            self.playerY = 0

        # 管線移動
        for uPipe, lPipe in zip(self.upperPipes, self.lowerPipes):
            uPipe['x'] += self.pipeVelX
            lPipe['x'] += self.pipeVelX

        # 當管線快到左側邊緣的時候，產生新的管線
        if 0 < self.upperPipes[0]['x'] < 5:
            newPipe = getRandomPipe()
            self.upperPipes.append(newPipe[0])
            self.lowerPipes.append(newPipe[1])

        # 當最前面的管線要出螢幕時刪除
        if self.upperPipes[0]['x'] < -PIPE_WIDTH:
            self.upperPipes.pop(0)
            self.lowerPipes.pop(0)


        isCrash_player = {
                'x': self.playerX,
                'y': self.playerY,
                'index': self.playerIndex
        }

        # 檢查碰撞
        isCrash = checkCrash(isCrash_player, self.upperPipes, self.lowerPipes)

        # 如果有發生碰撞，遊戲停止 terminal = True
        if isCrash:
            terminal = True
            self.__init__()
            reward = -1

        # 將所有角色都根據每個角色的座標畫到螢幕上
        SCREEN.blit(IMAGES['background'], (0,0))

        for uPipe, lPipe in zip(self.upperPipes, self.lowerPipes):
            SCREEN.blit(IMAGES['pipe'][0], (uPipe['x'], uPipe['y']))
            SCREEN.blit(IMAGES['pipe'][1], (lPipe['x'], lPipe['y']))

        SCREEN.blit(IMAGES['base'], (self.baseX, BASE_Y))
        SCREEN.blit(IMAGES['player'][self.playerIndex], (self.playerX, self.playerY))

        # 將目前的遊戲畫面產生一個 2D 畫面傳回
        image_data = pygame.surfarray.array3d(pygame.display.get_surface())
        pygame.display.update()
        FPSCLOCK.tick(FPS)

        return image_data, reward, terminal
