'''
Descripttion: 
Version: 1.0
Author: ZhangHongYu
Date: 2021-02-18 13:18:01
LastEditors: ZhangHongYu
LastEditTime: 2021-02-20 16:57:28
'''
from preprocess import read_data
from preprocess import features_eng
from sklearn.model_selection import train_test_split
from stacking import train_model
from stacking import evaluate_model
import numpy as np
import pandas as pd


if __name__ == '__main__':

    data = read_data()

    data = features_eng(data)

    
    submission_X = data[data['年份（年末）'] == 7].drop('是否高转送', axis=1)

    my_data = data[data['年份（年末）'] != 7]
    

    X, y = np.array(data.iloc[:, :-1]), np.array(data.iloc[:, -1])

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

    # 采用交叉验证法对初级学习器和次级学习器进行训练
    # train_model(X_train, y_train)

    # 评估学习器的性能
    evaluate_model(X_test, y_test)
