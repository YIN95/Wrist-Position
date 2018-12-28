# -*- coding: utf-8 -*-
import numpy as np
import cv2
import os
import pickle
import json

def saveImage(image, path):
    filenames = os.listdir(path + 'image/')
    print('data:', len(filenames))
    path = (path + 'image/' + '%d.jpg')%len(filenames)
    cv2.imwrite(path, image)

def saveLabel(label, path):
    # save as pickle
    # path = path + 'label/' + 'label.pkl'
    # with open(path, 'ab') as f:
    #     pickle.dump(label, f, pickle.HIGHEST_PROTOCOL)

    # save as json
    filename = len(os.listdir(path + 'image/'))-1
    path = path + 'label/' + 'label.json'
    save_dict = {filename: label}
    with open(path, 'a') as result_file:
        json.dump(save_dict, result_file)
        result_file.write('\n')
        result_file.close()


def saveData(image, label, path):
    saveImage(image, path)
    saveLabel(label, path)
