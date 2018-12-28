# -*- coding: utf-8 -*-
import argparse
import numpy as np
from getTarget import *


if __name__ == "__main__":
    print('Loading parameters ...')
    # Parsing
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera', type=int, default=1, help='camera number')
    parser.add_argument('--width', type=int, default=400, help='image width')
    parser.add_argument('--elliptical', type=int, default=12, help='elliptical kernel size')
    parser.add_argument('--gaussian', type=int, default=3, help='gaussian kernel size')
    parser.add_argument('--lower_skin', type=list, default=[0, 44, 70], help='lower bound of skin')
    parser.add_argument('--upper_skin', type=list, default=[30, 255, 255], help='upper bound of skin')
    parser.add_argument('--lower_target', type=list, default=[44, 0, 0], help='lower bound of target')
    parser.add_argument('--upper_target', type=list, default=[150, 100, 100], help='upper bound of target')

    args = parser.parse_args()
    getTarget(args)
    