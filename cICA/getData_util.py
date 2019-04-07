# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 05:33:26 2018

@author: Wenjie Yin

collect data for cICA
"""
import cv2 as cv
import os
import copy

class Camera():
    def __init__(self, savePath, label, cameraNum=0, Number=1000, delta=40):
        self.cameraNum = cameraNum
        self.savePath = savePath
        self.label = label
        self.cap = cv.VideoCapture(self.cameraNum)
        self.height = 0
        self.width = 0
        self.run = True
        self.collect = False
        self.count = 1
        self.Number = Number
        self.delta = delta
        self.x1 = 0
        self.x2 = 0
        self.y1 = 0
        self.y2 = 0

    def openCamera(self):
        if self.cap.isOpened():
            print('Camera opened')
            _, frame = self.cap.read()
            self.height, self.width = frame.shape[:2]
            print('Image Size: ', self.height, self.width)
            self.setDataRange()
            return True
        return False

    def closeCamera(self):
        print('Camera closed')
        self.run = False
        self.cap.release()
        cv.destroyAllWindows()

    def keyBoardController(self, key):
        if key > 0:
            if key & 0xff == ord('q') or key & 0xff == ord('Q'):
                self.closeCamera()
                return -1
            if key & 0xff == ord('y') or key & 0xff == ord('Y'):
                self.collect = True
                return 1
            if key & 0xff == ord('w') or key & 0xff == ord('W'):
                self.y1 -= 1
                self.y2 -= 1
                return 3
            if key & 0xff == ord('s') or key & 0xff == ord('S'):
                self.y1 += 1
                self.y2 += 1
                return 3
            if key & 0xff == ord('a') or key & 0xff == ord('A'):
                self.x1 -= 1
                self.x2 -= 1
                return 3
            if key & 0xff == ord('d') or key & 0xff == ord('D'):
                self.x1 += 1
                self.x2 += 1
                return 3
            if key & 0xff == ord('n') or key & 0xff == ord('N'):
                self.collect = False
                self.count = 1
                return 1

    def setDataRange(self, delta=40):
        centerX = self.width / 2
        centerY = self.height / 2

        x1 = int(centerX - delta / 2)
        x2 = int(centerX + delta / 2)
        y1 = int(centerY - delta / 2)
        y2 = int(centerY + delta / 2)
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        return x1, x2, y1, y2

    def collectData(self, img, frame, origin):
        if self.collect:
            if self.count <= self.Number:
                print("collect: ", self.count)
                imgname = self.label + str(self.count) + ".png"
                cv.imwrite(os.path.join(self.savePath, imgname), img)
                imgname = self.label + "global_" + str(self.count) + ".png"
                cv.imwrite(os.path.join(self.savePath, imgname), origin)
                self.count += 1

            else:
                self.collect = False
                imgname = self.label + "whole" + ".png"
                cv.imwrite(os.path.join(self.savePath, imgname), frame)
                self.closeCamera()

    def getCount(self):
        return self.count

    def getNumber(self):
        return self.Number

    def getHeight(self):
        return self.height

    def getWidth(self):
        return self.width

def getData_main():
    camera = Camera(savePath='D:\MyCode\cICA-Extracting-Correlated-Signals\Data\data_new',
                    label = 'new',
                    cameraNum=1,
                    Number=500,
                    delta=40)
    # whole = []
    camera.openCamera()
    while camera.run:
        _, frame = camera.cap.read()
        x1, x2, y1, y2 = camera.setDataRange()
        cv.rectangle(frame, (x1-4, y1-4), (x2+4, y2+4), (250, 0, 0), 2)
        cv.putText(frame, str(camera.count), (20, 40), 1, 2.5, (250, 0, 0), 2)
        img = frame[y1:y2, x1:x2]
        # whole = frame
        cv.imshow("origin", frame)
        camera.collectData(img, frame)
        key = cv.waitKey(1)
        camera.keyBoardController(key)

    # camera.collectData(whole)

def getData_whole_main():
    camera = Camera(savePath='D:\MyCode\cICA-Extracting-Correlated-Signals\Data\data_new',
                    label='new_',
                    cameraNum=0,
                    Number=500,
                    delta=150)
    camera.openCamera()
    range = 40
    camera.setDataRange(delta=range)
    half_range = int(range/2)
    while camera.run:
        _, frame = camera.cap.read()
        origin = copy.copy(frame)

        # size_whole, _ = frame.shape[:2]
        # cv.rectangle(frame, (x1 - int(size_whole/2), 0), (x2 + int(size_whole/2), size_whole-1), (0, 250, 0), 2)
        cv.rectangle(frame, (camera.x1 - half_range - 2, camera.y1 - half_range - 2), (camera.x2 + half_range + 2, camera.y2 + half_range + 2), (250, 0, 0), 2)

        cv.rectangle(frame, (110, 10), (camera.getWidth()-25, 40), (250, 0, 0), 2)
        current = int((camera.getWidth() - 135) * camera.getCount() / camera.getNumber() + 110)
        cv.rectangle(frame, (110, 10), (current, 40), (250, 0, 0), -1)

        cv.putText(frame, str(camera.count), (5, 40), 1, 2.5, (250, 0, 0), 2)
        img = frame[camera.y1:camera.y2, camera.x1:camera.x2]
        # whole = frame
        cv.imshow("origin", frame)
        camera.collectData(img, frame, origin)
        key = cv.waitKey(1)
        camera.keyBoardController(key)

if __name__ == '__main__':
    # getData_main()

    getData_whole_main()