import os
import numpy as np
import cv2
from glob import glob

# image size = (256, 256, 3)

images = glob('images/*.jpg')

for index, image in enumerate(images):
    # 이미지 불러오기
    img = cv2.imread(image)
    # 이미지 위아래 15% 잘라내기
    img = img[int(img.shape[0] * 0.15):int(img.shape[0] * 0.85), :, :]
    # 이미지 크기를 256x256으로 변경
    img = cv2.resize(img, (256, 256))
    # npy 파일로 저장
    np.save(f'np_images/{index}.npy', img)


