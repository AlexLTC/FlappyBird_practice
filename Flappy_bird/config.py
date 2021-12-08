# Hyper Parameters for Flappy Bird

class config(self):
    def __init__(self):
        self._GAME_NAME = 'bird'
        self._ACTIONS_NUM = 2
        self._GAMMA = 0.99
        self._OBSERVE = 10000.  # 訓練之前的時間步，需先觀察 10000 幀
        self._EXPLORE = 3000000.  # epslion 開始逐層變小
        self._FINAL_EPSLION = 0.0001
        self._INITIAL_EPSLION = 0.1
        self._REPLAY_MEMORY = 50000  # 最多記憶多少幀的訓練資料
        self._BATH_SIZE = 32
        self._FRAME_PER_ACTION = 1  # 每間隔多少時間完成一次有效的動作輸出

