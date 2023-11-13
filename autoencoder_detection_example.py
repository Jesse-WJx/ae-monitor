# -*- coding: utf-8 -*-
# 模型部分需要修改Model(inputs= , outputs= )
# monitor_a_sample return 计算的统计量
# cal_threshold return 控制限 
import pickle
import numpy as np
from sklearn.datasets import load_wine
from sklearn import preprocessing

from Detection_by_Autoencoder import DetectionbyAutoencoder as D_AE
from keras.models import load_model


from statistic import statistic_mahalanobis_distance as T2
from statistic import statistic_euclid_distance as SPE


def z_score(xtrain):
    """
    零均值单位方差标准化

    :param xtrain:训练集
    :param xtest:测试集
    :return:返回标准化之后的训练集和测试集
    Example:xtrain, xtest = z_score(xtrain, xtest)
    """
    row_xtrain, _ = np.shape(xtrain)
    xtrain_mean = np.mean(xtrain, axis=0)
    xtrain_std = np.std(xtrain, axis=0)
    # 标准化公式：x=(x-mean)/std
    xtrain = (xtrain - np.tile(xtrain_mean, (row_xtrain, 1))) / np.tile(xtrain_std, (row_xtrain, 1))
    return xtrain


if __name__ == '__main__':
    
    x1 = load_wine().data[0:59, :]
    x2 = load_wine().data[59:130, :]
    x3 = load_wine().data[130:, :]
    # print(type(x1)) array
    
    # StandardScaler = preprocessing.StandardScaler().fit(x1)  # 标准化
    # x1 = StandardScaler.transform(x1)
    # x2 = StandardScaler.transform(x2)
    # x3 = StandardScaler.transform(x3)

    x_mean = np.mean(x1, axis=0)
    x_std = np.std(x1, axis=0)
    x1 = z_score(x1)
    # print(x_mean, x_std)

    
    D_AE = D_AE(x1, [10, 8, 10])
    D_AE.offline_modeling()
    D_AE.get_offline_statistics()
    D_AE.save_model('autoencoder', 'encoder')
    D_AE.cal_threshold(0.99, use_kde=True)
    # for i in range(x2.shape[0]):
    #     ret = D_AE.monitor_a_sample(x2[i], print_info=False)
    #     ret_t.append(ret[0])
    #     ret_q.append(ret[1])
    # print(len(ret_t))
    D_AE.monitor_multi_sample(x2, print_info=False)

    autoencoder = load_model('autoencoder.h5')
    encoder = load_model('encoder.h5')
    
    feat = encoder.predict(x1)
    recon = autoencoder.predict(x1)
    sigma = np.linalg.inv(np.dot(feat.T, feat) / (x1.shape[0]-1))  # 训练数据算
    # print(sigma)
    # print(type(sigma))
    with open('sigma.pkl', 'wb') as f:  # 保存sigma
        pickle.dump(sigma, f)

    with open('x_mean.pkl', 'wb') as g:  # 保存训练集的均值与方差
        pickle.dump(x_mean, g)

    with open('x_std.pkl', 'wb') as h:
        pickle.dump(x_std, h)
 
    # 在线计算统计量
    # T2_online = np.zeros(x2.shape[0])
    # SPE_online = np.zeros(x2.shape[0])
    # for i in range(x2.shape[0]):   
    #     feat_new = encoder.predict(x2[i].reshape(1, -1))
    #     recon_new = autoencoder.predict(x2[i].reshape(1, -1))
    #     T2_online[i] = np.dot(feat_new, np.dot(sigma, feat_new.reshape(-1, 1)))
    #     residual = recon_new - x2[i]
    #     SPE_online[i] = np.dot(residual.reshape(1, -1), residual.reshape(-1, 1))[0,0]
    
    # print(SPE_online)
