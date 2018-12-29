# -*- coding: utf-8 -*-
import argparse
import numpy as np
from loadData import loadLabel, loadImage
from trainModel import train
import ast
import cv2

if __name__ == "__main__":
    print('\nLoading parameters', end=' ----------')
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataPath', type=str, default='./', help='Save directory')
    parser.add_argument('--dynamic', type=ast.literal_eval, default=False, help='Visualize data')
    parser.add_argument('--imgSize', type=int, default=28, help='image size')
    parser.add_argument('--trainingPercentage', type=int, default=80, help='What percentage of images to use as a training set.')
    parser.add_argument('--testingPercentage', type=int, default=20, help='What percentage of images to use as a testing set.')
    parser.add_argument('--modelPath', type=str, default='./', help='What path to save model')
    parser.add_argument('--modelName', type=str, default='model.h5', help='model name')

    args = parser.parse_args()
    print('Done')

    print('\nLoading Data', end=' ---------- ')
    X = loadImage(args.dataPath, args.dynamic, args.imgSize)
    Y = loadLabel(args.dataPath)
    print('Done')

    print('\nTraining ----------')
    model = train(X, Y, args)

    print('\nSaving', end=' ---------- ')
    model.save(args.modelPath + args.modelName) 
    print('Finish!')

    
    

    

