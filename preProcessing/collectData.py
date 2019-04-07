# -*- coding: utf-8 -*-
import numpy as np
import cv2
import os
import json
from collections import OrderedDict

def saveImage(image, path):
    '''
    save the image to the path
    '''
    filenames = os.listdir(path + 'image/')
    print('data:', len(filenames))
    path = (path + 'image/' + '%d.jpg')%len(filenames)
    cv2.imwrite(path, image)

def saveLabel(label, path):
    '''
    save the label of the image
    '''
    # save as pickle
    # path = path + 'label/' + 'label.pkl'
    # with open(path, 'ab') as f:
    #     pickle.dump(label, f, pickle.HIGHEST_PROTOCOL)

    # save as json
    filename = len(os.listdir(path + 'image/'))-1
    listdir = os.listdir(path + 'label/')
    path = path + 'label/' + 'label.json'
    labelLog = {}

    if filename > 0:
        with open(path,'r') as load_f:
            # save the data with order
            labelLog = json.load(load_f, object_pairs_hook=OrderedDict) 
        labelLog[filename] = label
        load_f.close()

        with open(path, 'w') as result_file:
            json.dump(labelLog, result_file, indent = 4)
            # result_file.write('\n')
            result_file.close()
    else:
        # save the first data
        with open(path, 'w') as result_file:
            labelLog[filename] = label
            json.dump(labelLog, result_file)
            result_file.close()

def saveData(image, label, path):
    if (label[0]==0 and label[1]==0):
        pass
    else:
        saveImage(image, path)
        saveLabel(label, path)
