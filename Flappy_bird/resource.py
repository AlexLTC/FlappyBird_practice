import pygame
import sys

def getHitmask(image):
    # 根據影像的 alpha，獲得遮色片
    mask = []
    for x in range(image.get_width()):
        mask.append([])
        for y in range(image.get_height()):
            mask[x].append(bool(image.get_at((x, y))[3])) # image.get_at((x, y))[3]) 代表 alpha(通道)，也就是 RGBA 的 A

    return mask

# 載入各種資源的函數
def load():
    # 小鳥在不同狀態下的圖片
    PLAYER_PATH = (
            'assets/sprites/redbird-upflap.png',
            'assets/sprites/redbird-midflap.png',
            'assets/sprites/redbird-downflap.png'
    )

    # 背景圖地址
    BACKGROUND_PATH = 'assets/sprites/background-black.png'

    # 管線圖片所在地址
    PIPE_PATH = ('assets/sprites/pipe-green.png')

    IMAGES, SOUNDS, HITMASKS = {}, {}, {}

    # 載入成績數字所需的圖片
    IMAGES['numbers'] = (
            pygame.image.load('assets/sprites/0.png').convert_alpha(),
            pygame.image.load('assets/sprites/1.png').convert_alpha(),
            pygame.image.load('assets/sprites/2.png').convert_alpha(),
            pygame.image.load('assets/sprites/3.png').convert_alpha(),
            pygame.image.load('assets/sprites/4.png').convert_alpha(),
            pygame.image.load('assets/sprites/5.png').convert_alpha(),
            pygame.image.load('assets/sprites/6.png').convert_alpha(),
            pygame.image.load('assets/sprites/7.png').convert_alpha(),
            pygame.image.load('assets/sprites/8.png').convert_alpha(),
            pygame.image.load('assets/sprites/9.png').convert_alpha()
    )

    # 載入地面的圖片
    IMAGES['base'] = pygame.image.load('assets/sprites/base.png').convert_alpha()

    # 載入音效檔（不同系統音效檔不同）
    if 'win' in sys.platform:
        sound.Ext = '.wav'
    else:
        soundExt = '.ogg'

    SOUNDS['die'] = pygame.mixer.Sound('assets/audio/die' + soundExt)
    SOUNDS['hit'] = pygame.mixer.Sound('assets/audio/hit' + soundExt)
    SOUNDS['point'] = pygame.mixer.Sound('assets/audio/point' + soundExt)
    SOUNDS['swoosh'] = pygame.mixer.Sound('assets/audio/swoosh' + soundExt)
    SOUNDS['wing'] = pygame.mixer.Sound('assets/audio/wing' + soundExt)

    # 載入背景圖
    IMAGES['background'] = pygame.image.load(BACKGROUND_PATH).convert()

    # 載入小鳥的圖片
    IMAGES['player'] = (
            pygame.image.load(PLAYER_PATH[0]).convert_alpha(), 
            pygame.image.load(PLAYER_PATH[1]).convert_alpha(),
            pygame.image.load(PLAYER_PATH[2]).convert_alpha()
    )

    # 載入水管
    IMAGES['pipe'] = (
            pygame.transform.rotate(pygame.image.load(PIPE_PATH).convert_alpha(), 180),
            pygame.image.load(PIPE_PATH).convert_alpha()
    )

    # 獲得水管遮色片
    # 遮色片：將影像中的主體從整個影像擷取出來的技術，方便將主體與其他物件合成到一起; 遮色騙會以 boolean 的方式儲存
    HITMASKS['pipe'] = (
            getHitmask(IMAGES['pipe'][0]),
            getHitmask(IMAGES['pipe'][1])
    )

    # 玩家的遮色片
    HITMASKS['player'] = (
            getHitmask(IMAGES['player'][0]),
            getHitmask(IMAGES['player'][1]),
            getHitmask(IMAGES['player'][2])
    )

    return IMAGES, SOUNDS, HITMASKS

