# -*- coding: utf-8 -*-
import argparse
import cv2
import imutils
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"   
sys.path.append('../preProcessing/')
from getTarget import maskProcess
from keras.models import load_model
from skimage.measure import label
import numpy as np
 

if __name__ == "__main__":
    print('\nLoading parameters', end=' ----------')
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera', type=int, default=0, help='camera number')
    parser.add_argument('--modelPath', type=str, default='./', help='What path to save model')
    parser.add_argument('--modelName', type=str, default='model.h5', help='model name')
    parser.add_argument('--imgSize', type=int, default=28, help='image size')

    parser.add_argument('--width', type=int, default=400, help='image width')
    parser.add_argument('--elliptical', type=int, default=8, help='elliptical kernel size')
    parser.add_argument('--gaussian', type=int, default=3, help='gaussian kernel size')
    parser.add_argument('--lower_skin', type=list, default=[0, 44, 70], help='lower bound of skin')
    parser.add_argument('--upper_skin', type=list, default=[30, 255, 255], help='upper bound of skin')
    parser.add_argument('--lower_label', type=list, default=[60, 200, 200], help='lower bound of target')
    parser.add_argument('--upper_label', type=list, default=[100, 245, 240], help='upper bound of target')
    parser.add_argument('--dataPath', type=str, default='./', help='Save directory')

    args = parser.parse_args()
    print('Done')

    print('\nLoading Model', end=' ---------- ')
    model = load_model(args.modelPath + args.modelName)
    print('Done')

    print('\Testing ----------')
    if args.camera >= 0:
        camera = cv2.VideoCapture(args.camera)
        # keep looping over the frames in the video
        while True:
            # grab the current frame
            _, frame = camera.read()
            frame = imutils.resize(frame, width = args.width)
            minsize = min(frame.shape[0], frame.shape[1])
            frame = frame[0:minsize, 0:minsize, :]
            onprocess, img = maskProcess(frame, args)
            img = cv2.resize(img, (args.imgSize, args.imgSize), interpolation=cv2.INTER_CUBIC)
            cv2.imshow("images-3", img)
            # print()
            
            inputData = np.zeros((1, 1, args.imgSize, args.imgSize))
            inputData[0, 0, :, :] = img
            result = model.predict(inputData/255.)
            print(result[0])

            cv2.circle(frame, (result[0][0], result[0][1]), 5, (0,255,), -1)
            cv2.imshow("images-4", frame)

            key = cv2.waitKey(1)
            if key & 0xFF == ord("q"):
                break

    print('Finish!')

    
    

    

