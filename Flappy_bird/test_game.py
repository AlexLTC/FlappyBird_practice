import matplotlib.pyplot as plt
from IPython.display import display, clear_output
import numpy as np

import create_game

game = create_game.GameState()

fig = plt.figure()
axe = fig.add_subplot(111)
dat = np.zeros((10, 10))
img = axe.imshow(dat)


for i in range(100):
    clear_output(wait = True)

    # input_actions = [0, 1] 代表按著按鍵不放，所以圖像中鳥會往上
    image_data, reward, terminal = game.frame_step([0, 1])
    
    image = np.transpose(image_data, [1, 0, 2])
    img.set_data(image)
    img.autoscale()
    display(fig)
