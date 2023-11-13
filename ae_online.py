# -*- coding: utf-8 -*-
# 模型部分需要修改Model(inputs= , outputs= )
# monitor_a_sample return 计算的统计量
# cal_threshold return 控制限 
import pickle
import numpy as np
from sklearn.datasets import load_wine
from sklearn import preprocessing

# from Detection_by_Autoencoder import DetectionbyAutoencoder as D_AE
from keras.models import load_model


if __name__ == '__main__':
    
    x1 = load_wine().data[0:59, :]
    x2 = load_wine().data[59:130, :]
    # x3 = load_wine().data[130:, :]
    # print(type(x1)) array
    
    # StandardScaler = preprocessing.StandardScaler().fit(x1)
    # x1 = StandardScaler.transform(x1)
    # x2 = StandardScaler.transform(x2)
    # x3 = StandardScaler.transform(x3)
    autoencoder = load_model('autoencoder.h5')
    encoder = load_model('encoder.h5')

    with open('sigma.pkl', 'rb') as f:
        sigma = pickle.load(f)

    with open('x_mean.pkl', 'rb') as g:
        x_mean = pickle.load(g)
    
    with open('x_std.pkl', 'rb') as h:
        x_std = pickle.load(h)
    
    # print(x_mean, x_std)
    
    # print(sigma)
   
    T2_online = np.zeros(x2.shape[0])
    SPE_online = np.zeros(x2.shape[0])
    for i in range(x2.shape[0]):   
        x2[i] = (x2[i] - x_mean) / x_std
        feat_new = encoder.predict(x2[i].reshape(1, -1))
        recon_new = autoencoder.predict(x2[i].reshape(1, -1))
        T2_online[i] = np.dot(feat_new, np.dot(sigma, feat_new.reshape(-1, 1)))
        residual = recon_new - x2[i]
        SPE_online[i] = np.dot(residual.reshape(1, -1), residual.reshape(-1, 1))[0,0]

