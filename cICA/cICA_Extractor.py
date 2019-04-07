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

from cICA_util import CICA
import numpy as np
import matplotlib.pyplot as plt
import math


showData = True
DirectAutoCorrelation = True

def cICAmain():
    cica = CICA(fileDir='D:\MyCode\cICA-Extracting-Correlated-Signals\Data',
                fileName='data_new',
                dataName='new_',
                dataType='.png',
                indX=([0, 40]),
                indY=([0, 40]),
                Len=410,
                fs=30)

    Data, t = cica.videoSeg(show=True)
    cica.plotData(Data, t, show=True, title="Data_Original")
    Data_BH = cica.slidingBHWindow(Data, stepSize=int(cica.fs*0.5), windowSize=int(cica.fs*1.5),
                                lowCutFrq=0.25, highCutFrq=5)
    cica.plotData(Data_BH, t, show=True, title="Data_BH")

    '''
    Method: Direct AutoCorrelation, which is very useful!
    '''
    if DirectAutoCorrelation:
        PeriodN = np.zeros(3)
        for i in range(3):
            PeriodN[i] = cica.autoCorr2Freq(Data_BH[i, :], periodMin=15, show=False)

        if not math.isnan(np.nanmean(PeriodN)):
            periodN_Best = int(np.nanmean(PeriodN))

            # Finding the best delay
            periodN = periodN_Best
            Delay = np.linspace(0, periodN, periodN)
            Data_Ref = np.array([np.cos(2 * np.pi * ((t - i / cica.fs) * (cica.fs / periodN))) for i in Delay])
            cica.plotData(Data_Ref, t, show=False, title="Ref")
            cica.autoCorr2Freq(Data_Ref[1, :], periodMin=15, show=False)
            tmp = np.array([np.dot(Data_BH, Data_Ref[i, :]) for i in range(0, Data_Ref.shape[0])])
            cica.plotData(tmp, range(0, 3), show=True, title="tmp 1")

            tmp = [np.sum(np.dot(Data_BH, Data_Ref[i, :]), axis=0) for i in range(0, Data_Ref.shape[0])]
            Delay_Best = int(Delay[np.argmax(tmp)])
            Data_Ref_ = np.cos(2 * np.pi * ((t - Delay[np.argmax(tmp)] / cica.fs) * (cica.fs / periodN)))
            Data_Ref_ = Data_Ref_ / np.linalg.norm(Data_Ref_) * np.linalg.norm(Data, axis=1).mean()
            cica.plotData(np.concatenate((Data_BH, Data_Ref_.reshape(1, len(Data_Ref_))), axis=0), t,
                          show=True, title="data ref")

            # Finding the best periodic component
            w, y = cica.constrainedICA(Data_BH, Data_Ref_, mu=1, maxIter = 300)
            xw1 = np.dot(w.T, Data_BH)
            cica.plotData(np.concatenate((Data_BH, Data_Ref_.reshape(1, len(Data_Ref_))), axis=0), t,
                          show=True, title="Finding the Best periodic component")

            if showData:
                plt.plot(t, -xw1.T, color='grey', linewidth=4)
                cica.autoCorr2Freq(xw1[0, :], periodMin=15, show=True)
                plt.title("auto-correlation of best periodic component")

            # calculating an average period
            xw = xw1[0, Delay_Best:Delay_Best + periodN_Best * math.floor(xw1.size / periodN_Best)]
            sizeX = math.floor(xw.size / periodN_Best) * periodN_Best
            xw = xw[0: sizeX]
            xw = np.reshape(xw, (math.floor(xw.size / periodN_Best), periodN_Best))
            xwmean = np.nanmean(xw, axis=0)
            cica.plotData(np.concatenate((xw, xwmean.reshape(1, periodN_Best)), axis=0), range(periodN_Best), show=True, title="average period")
            plt.plot(range(periodN_Best), xwmean, color='grey', linewidth=4)

    plt.show()

def example():
    cica = CICA(fileDir='D:\MyCode\cICA-Extracting-Correlated-Signals\Data',
                fileName='data_new',
                dataName='new',
                dataType='.png',
                indX=([0, 40]),
                indY=([0, 40]),
                Len=310,
                fs=30)
    # example
    from numpy import pi
    N = 3000
    ts = 1e-2
    k = np.linspace(0,ts*N,N)
    f1 = 0.061/ts
    f2 = 0.54/ts
    f3 = 0.28/ts
    S = np.array([np.sin(2*pi*f1*ts*k) + 6*np.cos(2*pi*200*ts*k),
      np.cos(2*pi*f2*ts*k),
      np.cos(2*pi*f3*ts*k + 2),
      np.random.randn(N),
      np.random.randn(N)
      ])
    T1 = 1/(f1*ts)
    T2 = 1/(f2*ts)
    plt.plot(S.T)

    A = np.random.randn(5,5)
    X = np.dot(A,S)
    plt.figure()
    plt.plot(X.T)
    ref = S[0,:]
    ref = ref/np.linalg.norm(ref)*2.0

    from sklearn.decomposition import PCA, FastICA
    ica = FastICA(n_components=5)
    S_ = ica.fit_transform(X.T)
    plt.figure()
    plt.plot(S_)

    X_prewh = X
    X, white, dewhite = cica.prewhiten(X_prewh)
    plt.figure()
    plt.plot(X.T)
    plt.plot(ref)
    w, y = cica.constrainedICA(X, ref, mu=1, lam=1,maxIter = 100,overValue=1e-7, threshold=1.75)
    plt.show()

if __name__ == '__main__':
    cICAmain()
    # example()