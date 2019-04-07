# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 02:33:42 2018

@author: Jingjing Luo, Wenjie Yin

cICA:
    X: input matrix, array?

Reference:
    [1] Wei Lu, Jagath C. Rajapakse: ICA with reference. ICA 2001
    [2] Zhi-Lin Zhang, Morphologically Constrained ICA for Extracting Weak Temporally
        Correlated Signals, Neurocomputing 71(7-9) (2008) 1669-1679
"""
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.signal import hamming
from sklearn.preprocessing import normalize
from scipy.linalg import eigh
from scipy.signal import find_peaks
import cv2 as cv
from matplotlib import cm

class CICA():
    def __init__(self, fileDir, fileName, dataName, dataType='.png',
                 indX=([10, 30]), indY=([10, 30]), Len=500, fs=30):
        self.fileDir = fileDir
        self.fileName = fileName
        self.dataName = dataName
        self.dataType = dataType
        self.indX = indX
        self.indY = indY
        self.Len = Len
        self.fs = fs
        self.hsv_ = False

    def loadData(self, show=True):
        '''
        load the image data from the file

        :return: Data
        '''
        Data = np.zeros([self.indX[1] - self.indX[0], self.indY[1] - self.indY[0], 3, self.Len])
        print(Data.shape)
        for i in range(1, self.Len+1):
            filePath = os.path.join(self.fileDir, self.fileName, self.dataName + str(i) + self.dataType)
            image = plt.imread(filePath)
            temp = image[self.indX[0]:self.indX[1], self.indY[0]:self.indY[1], :]
            Data[:, :, :, i-1] = temp
            if show:
                if i == 1:
                    plt.figure()
                    plt.imshow(image[self.indX[0]:self.indX[1], self.indY[0]:self.indY[1]])
                    plt.title("The origin data")
        return Data

    def videoSeg(self, show=True):
        '''
        TODO

        :return: Data, t
        '''
        Data_ = self.loadData(show=show)
        temp = np.mean(Data_, axis=1)
        Data = np.mean(temp, axis=0)
        t = 0
        for i in range(0, 3):
            if self.hsv_ == True:
                Data[i, :] = (Data[0, :] - Data[0, :].mean())
            if self.hsv_ == False:
                Data[i, :] = (Data[i, :] - Data[i, :].mean())
            t = np.linspace(0, Data.shape[1] / self.fs, Data.shape[1])

        return Data, t

    def plotData(self, Data, t, show, title="data"):
        r, c = Data.shape
        colors = plt.cm.jet(np.linspace(0, 1, r))
        if r >= 6:
            r = 0
        if show:
            plt.figure()
            plt.subplot(r+1, 1, 1)

            for i in range(0, Data.shape[0]):
                plt.plot(t, Data[i, :], c=colors[i, :])
            plt.legend(range(0, Data.shape[0]))
            plt.title(title)

            for i in range(r):
                plt.subplot(r + 1, 1, i + 2)
                plt.plot(t, Data[i, :], c=colors[i, :])


    def bandpass(self, data, lowCutFrq=0.5, highCutFrq=2.5, butterOrderN=5):
        nyp = 0.5 * self.fs
        lowCut = lowCutFrq / nyp
        highCut = highCutFrq / nyp
        [b, a] = butter(butterOrderN, ([lowCut, highCut]), btype='bandpass')
        data = filtfilt(b, a, data)
        return data

    def slidingBHWindow(self, Data, stepSize=20, windowSize=100, lowCutFrq=0.5, highCutFrq=2.5):
        Hwindow = hamming(windowSize)
        Data_BH = np.zeros(Data.shape)
        for i in range(0, Data.shape[1] - windowSize, stepSize):
            data_ = Data[:, i:i + windowSize]
            data_ = data_ - data_.mean(axis=1).reshape((-1, 1))
            data = self.bandpass(data_, lowCutFrq, highCutFrq)
            Data_BH[:, i:i + windowSize] = Data_BH[:, i:i + windowSize] + data * Hwindow
        return Data_BH

    def autoCorr2Freq(self, data, periodMin, show):
        n = data.size
        if n == 1:
            data = data.transpose()
            n = data.size
            print('Warning!!! Input consists only one elements, not valid for frequency calculating')
            return

        data_norm = data - np.mean(data)
        result = np.correlate(data_norm, data_norm, mode='same')

        # checking for the periodic property of auto-correlation
        autocorrelation = result[n//2 + 1:] / (data.var() * np.arange(n-1, n//2, -1))
        lag = np.abs(autocorrelation).argmax() + 1
        r = autocorrelation[lag - 1]
        peaks, _ = find_peaks(autocorrelation, width=int(periodMin / 5), distance=periodMin, height=0.25)

        if show:
            plt.figure()
            plt.plot(autocorrelation)
            plt.plot(peaks, autocorrelation[peaks], "x")
            plt.title("auto-correlation peaks")

        # getting period of time-points of the periodic auto-correlation
        if r > 0.5:
            indexs = np.diff(peaks)
            tmp = np.array([i for i, j in enumerate(np.diff(indexs)) if np.abs(j) < int(periodMin / 5)])
            if len(tmp):
                periodN = int(np.sum(indexs[tmp.transpose()] / len(tmp)))
                print('Auto-correlation is periodic, period of time points is: ', periodN)
            else:
                print('Warning!!! Auto-correlation is non-periodic, peak amount is <2')
                periodN = np.nan
        else:
            print('Warning!!! Auto-correlation is non-periodic')
            periodN = np.nan
        return periodN

    def constrainedICA(self, X, ref, mu=1, lam=1, maxIter=10, overValue=1e-7, threshold=1.75):
        ICnum, IClen = X.shape
        learningRate = 0.2
        gamma = 1

        # Defining initial parameters
        w = np.random.randn(ICnum, 1)
        w = w / np.linalg.norm(w)
        oldw = w
        flag = 1
        loop = 1

        # comput the covarience matrix Rxx
        Rxx = np.dot(X, X.T) / IClen  # np.cov(X)

        while flag == 1:
            # output estimation at current iteration
            y = np.matmul(w.T, X)

            # calculate the 1st-order deviationg of the Lagarange Function
            std_y = y.std()
            v_Gaus = np.random.normal(0, std_y, IClen)
            rou = np.mean(np.log10(np.cosh(y)) - np.log10(np.cosh(v_Gaus)))
            L1 = np.sign(rou) * np.dot(X, np.tanh(y).T) / IClen\
                 - mu * np.dot(X, (y - ref).T) / IClen\
                 - lam * np.dot(X, y.T) / IClen

            # related to the 2nd-order deviation of the Lagarange Function
            L2 = np.sign(rou) * (1 - np.power(np.tanh(y), 2)).mean() - mu - lam

            # update of the weight vector
            w = w - learningRate * np.dot(np.linalg.inv(Rxx), L1) / L2
            w = w / np.linalg.norm(w)

            # updata o the parameters
            thr = threshold * (1 - np.exp(-loop))
            g = np.mean(np.power(y - ref, 2)) - thr / IClen
            mu = np.max([0, mu + gamma * g])
            h = np.mean(np.power(y, 2)) - 1  # corresponds to the equality constraint
            lam = lam + gamma * h

            # decide whether the algorithm has convered
            wchange = 1 - np.abs(np.dot(w.T, oldw))
            print(loop, ' iterations: change in w is ', wchange, ', w is ', w.T)
            if wchange < overValue:
                print('Converged after ', loop, ' iteration.\n')
                flag = 0
            if loop >= maxIter:
                print('After ', loop, ' iteration, still cannot converge.\n')
                flag = 0

            # update the rest of the parameters
            oldw = w
            loop = loop + 1

        y = np.matmul(w.T, X)
        print('End of cICA algorith. \n')
        return w, y

    def diagsqrts(this, w):
        """
        Returns direct and inverse square root normalization matrices
        """
        Di = np.diag(1. / (np.sqrt(w) + np.finfo(float).eps))
        D = np.diag(np.sqrt(w))
        return D, Di

    def prewhiten(this, X2d):
        """ data Whitening
        *Input
        X2d : 2d data matrix of observations by variables
        ICnum: Number of components to retain
        *Output
        Xwhite : Whitened X
        white : whitening matrix (Xwhite = np.dot(white,X))
        dewhite : dewhitening matrix (X = np.dot(dewhite,Xwhite))
        """
        Nrow, Ncol = X2d.shape
        if Nrow > Ncol:
            X2d = X2d.transpose()
            Nrow, Ncol = X2d.shape

        X2d_norm = normalize(X2d, axis=1, norm='l2')
        Cov = np.dot(X2d_norm, X2d_norm.T) / (Nrow - 1)
        w, u = eigh(Cov)
        D, Di = this.diagsqrts(w)
        white = np.dot(Di, u.T)
        x_white = np.dot(white, X2d_norm)
        dewhite = np.dot(u, D)
        plt.figure()
        plt.plot(x_white.T)
        return (x_white, white, dewhite)

    def weightedCorr(self, Data, xw, w):
        Data_b = Data[:, :, 0, :]
        Data_g = Data[:, :, 1, :]
        Data_r = Data[:, :, 2, :]

        cor_b = np.zeros([Data.shape[0], Data.shape[1]])
        cor_g = np.zeros([Data.shape[0], Data.shape[1]])
        cor_r = np.zeros([Data.shape[0], Data.shape[1]])

        for i in range(Data.shape[0]):
            for j in range(Data.shape[1]):
                data_ = Data_b[i, j, :]
                temp = np.corrcoef(xw, data_)
                cor_b[i, j] = temp[0, 1]

                data_ = Data_g[i, j, :]
                temp = np.corrcoef(xw, data_)
                cor_g[i, j] = temp[0, 1]

                data_ = Data_r[i, j, :]
                temp = np.corrcoef(xw, data_)
                cor_r[i, j] = temp[0, 1]

        cor = cor_b * w[0] + cor_g * w[1] + cor_r * w[2]

        return cor

    def draw_heatmap(self, data):
        cmap = cm.get_cmap('rainbow', 1000)
        figure = plt.figure(facecolor='w')
        ax = figure.add_subplot(1, 1, 1)
        # print(data.shape)
        data = data.transpose()
        vmax = data.max()
        vmin = data.min()
        map = ax.imshow(data, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
        cb = plt.colorbar(mappable=map, cax=None, ax=None, shrink=0.5)
        plt.show()

