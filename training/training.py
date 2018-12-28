# -*- coding: utf-8 -*-
import argparse
import numpy as np
from getData import loadLabel, loadImage
import ast
import cv2

if __name__ == "__main__":
    print('Loading parameters ...')
    # Parsing
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataPath', type=str, default='./', help='Save directory')
    parser.add_argument('--dynamic', type=ast.literal_eval, default=False, help='Visualize data')
    parser.add_argument('--imgSize', type=int, default=128, help='image size')
    args = parser.parse_args()

    print('Loading image ...')
    X = loadImage(args.dataPath, args.dynamic, args.imgSize)

    print('Loading label ...')
    Y = loadLabel(args.dataPath)
    
    