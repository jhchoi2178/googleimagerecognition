import os
import numpy as np
import pandas as pd 
from PIL import Image
from cv2 import resize
import matplotlib.pyplot as plt

dir_image_train = "/data_train/"
image_train = os.listdir(dir_image_train)
image_train_resized = []
cnt = 0
MAX = 1000
for filename in image_train:
	im = np.array(Image.open(image_train+filename).resize((256,256),Image.LANCZOS))
	all_images_resized.append(im)
	cnt = cnt+1;
	if(cnt>MAX) break;

fig = plt.figure(figsize = (16, 32))
for index, im in zip(range(1, len(all_images_resized)+1), all_images_resized):
    fig.add_subplot(10, 5, index)
    plt.title(filename)
    plt.imshow(im)   