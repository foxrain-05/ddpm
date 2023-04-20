import os
import numpy as np
import cv2
from glob import glob

# image size = (256, 256, 3)

def set_images():
    images = glob('images/*.jpg')

    for index, image in enumerate(images):
        img = cv2.imread(image)
        img = img[int(img.shape[0] * 0.15):int(img.shape[0] * 0.85), :, :]
        img = cv2.resize(img, (256, 256))
        np.save(f'np_images/{index}.npy', img)



if __name__ == '__main__':
    set_images()
