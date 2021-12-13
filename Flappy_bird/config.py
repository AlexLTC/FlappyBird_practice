# Hyper Parameters for Flappy Bird
from easydict import EasyDict as edict

__C = edict()
cfg = __C

__C._GAME_NAME = 'bird'
__C._ACTIONS_NUM = 2
__C._GAMMA = 0.99
__C._OBSERVE = 10000.  # 訓練之前的時間步，需先觀察 10000 幀
__C._EXPLORE = 3000000.  # epslion 開始逐層變小
__C._FINAL_EPSILON = 0.0001
__C._INITIAL_EPSILON = 0.1
__C._REPLAY_MEMORY = 50000  # 最多記憶多少幀的訓練資料
__C._BATCH_SIZE = 32
__C._FRAME_PER_ACTION = 1  # 每間隔多少時間完成一次有效的動作輸出

