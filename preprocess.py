'''
Descripttion: 
Version: 1.0
Author: ZhangHongYu
Date: 2021-02-18 13:15:08
LastEditors: ZhangHongYu
LastEditTime: 2021-02-20 21:58:11
'''
import pandas as pd
import numpy as np
import os
from sklearn import preprocessing

# 数据存放目录定义
data_root = '/home/macong/project/A题全部数据/'

def read_data():
    data1 = pd.read_csv(data_root+'基础数据.csv', encoding='GB2312')
    data2 = pd.read_csv(data_root+'年数据.csv', encoding='GB2312')
    # print(data2)
    # reader3 = pd.read_table(
    #     os.path.join(data_root, '日数据.csv'),
    #     encoding='GB2312',
    #     sep=',',
    #     iterator=True)
    # chunks = []
    # loop = True
    # chunkSize = 10000
    # while loop:
    #     try:
    #         chunk = reader3.get_chunk(chunkSize)
    #         print("**********")
    #         print(chunk)
    #         print("*********")
    #         chunks.append(chunk)
    #     except StopIteration:
    #         loop = False
    #         print("Iteration is stopped.")
    # data = pd.concat(chunks, ignore_index=True)
    # 结合基本数据和年数据，暂时不考虑日数据
    combined_data = pd.merge(data2, data1, how="outer", on="股票编号")
    labels = combined_data['是否高转送'].to_list()
    # 我们根据上一年的特征预测下一年是否高送转，故标签是下一年的
    for i in range(len(labels)-1):
        labels[i] = labels[i+1]
    combined_data['是否高转送'] = pd.Series(labels)
    return combined_data


def features_eng(data):
    #  获得每个特征的缺失信息
    null_info = data.isnull().sum(axis=0)
    #  丢弃缺失值多于30%的特征
    features = [k for k, v in dict(null_info).items() if v < data.shape[0]* 0.3]
    data = data[features]
    null_info = data.isnull().sum(axis=0)

    # 选去出需要填补缺失值的特征
    features_fillna = [k for k, v in dict(null_info).items() if v > 0]
    # 缺失值填充，将列按出现频率由高到低排序，众数即第一行，inplace表示原地修改

    # 用众数对缺失值进行填补
    data.loc[:,  features_fillna] = data[features_fillna].fillna(
        data[features_fillna].mode().iloc[0]
    )

    print(data.shape)
    # 字符独热编码，数值归一化
    for col in data.columns:
        if col == '是否高转送':  # 跳过标签列
            continue
        if str(data[col].dtype) == 'object':
            # 字符->数值
            data.loc[:, col] = pd.factorize(
                data[col])[0]
            # 获取one-hot编码
            dummies_df = pd.get_dummies(data[col], prefix=str(col))
            data = data.drop(col, axis=1)
            data = data.join(dummies_df)
        else:
            # 对数值特征归一化
            scaler = preprocessing.StandardScaler().fit(
                np.array(data[col]).reshape(-1, 1))
            data.loc[:,  col] = scaler.transform(np.array(data[col]).reshape(-1, 1))
    return data
