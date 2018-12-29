import json
import numpy as np
import os
import cv2

def loadLabel(path):
    path = path + 'label/' + 'label.json'
    with open(path,'r') as load_f:
        load_dict = json.load(load_f)
    dataLabel = []
    for i in range(len(load_dict)):
        dataLabel.append(load_dict[str(i)])
    
    dataLabel = np.array(dataLabel)
    return dataLabel

def loadImage(path, dynamic, imgSize):
    path = path + 'image/' 
    filenames = os.listdir(path)
    dim = len(filenames)
    dataImage = np.ones([dim, 1, imgSize, imgSize])
    for i in range(dim):
        imgPath = path + str(i) + '.jpg'
        img = cv2.imread(imgPath)
        img = cv2.resize(img, (imgSize, imgSize), interpolation=cv2.INTER_CUBIC)

        if dynamic:
            cv2.imshow('imgData', img)
            key = cv2.waitKey(1)

        img = img/255.0
        dataImage[i, 0, :, :] = img[:, :, 0]

    return dataImage