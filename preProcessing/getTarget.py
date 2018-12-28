# -*- coding: utf-8 -*-
import numpy as np
import imutils
import argparse
import cv2
from skimage.measure import label

def largestConnectComponent(frame):
    '''
    compute largest Connect component of an labeled image
    '''
    bw_img = frame.copy()
    labeled_img, num = label(bw_img, neighbors=4, background=0, return_num=True)    

    max_label = 0
    max_num = 0
    for i in range(0, num): 
        index = np.where(labeled_img==i)
        c_index = index[0][0]
        l_index = index[1][0]        
        if (frame[c_index][l_index]) == 255:
            if (np.sum(labeled_img == i) > max_num):
                max_num = np.sum(labeled_img == i)
                max_label = i
    lcc = (labeled_img == max_label)

    # cv2.imshow("images-3", labeled_img)
    bw_img[np.where(labeled_img!=max_label)]=0
    bw_img[np.where(labeled_img==max_label)]=255
    
    index = np.where(bw_img==255)
    c_index = index[0][0]
    l_index = index[1][0]
    if (frame[c_index][l_index]) == 0:
        bw_img[np.where(labeled_img!=max_label)]=255
        bw_img[np.where(labeled_img==max_label)]=0
        
    return bw_img

def maskProcess(frame, args):
    '''
    use mask to get the interest area
    '''
    lower_skin = np.array(args.lower_skin)
    upper_skin = np.array(args.upper_skin)
    converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    skinMask = cv2.inRange(converted, lower_skin, upper_skin)

    # apply a series of erosions and dilations to the mask
    # using an elliptical kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (args.elliptical, args.elliptical))
    skinMask = cv2.erode(skinMask, kernel, iterations = 2)
    skinMask = cv2.dilate(skinMask, kernel, iterations = 2)

    # blur the mask to help remove noise, then apply the
    # mask to the frame
    skinMask = cv2.GaussianBlur(skinMask, (args.gaussian, args.gaussian), 0)
    skin = cv2.bitwise_and(frame, frame, mask = skinMask)

    # convert to gray-scale image
    gray = cv2.cvtColor(skin, cv2.COLOR_BGR2GRAY)

    # convert to binary image
    _,binary = cv2.threshold(gray,2,255,cv2.THRESH_BINARY)
    dilated = cv2.dilate(binary,kernel)

    largest = largestConnectComponent(dilated)
    
    # show the skin in the image along with the mask
    cv2.imshow("images-1", np.hstack([frame, skin]))
    cv2.imshow("images-2", np.hstack([gray, dilated, largest]))
    # if the 'q' key is pressed, stop the loop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        return False

    return True

def getTarget(args):
    '''
    get the area of skin and position of target
    '''
    print("camera number:", args.camera)
    # check whether use video, if not, open camera
    if args.camera >= 0:
        camera = cv2.VideoCapture(args.camera)
    
        # keep looping over the frames in the video
        while True:
            # grab the current frame
            _, frame = camera.read()
            frame = imutils.resize(frame, width = args.width)

            onprocess = maskProcess(frame, args)
            if not onprocess:
                break
                        
        # cleanup the camera and close any open windows
        camera.release()
        cv2.destroyAllWindows()
        
