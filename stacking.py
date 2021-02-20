'''
Descripttion: Stacking
Version: 1.0
Author: ZhangHongYu
Date: 2021-02-18 13:11:38
LastEditors: ZhangHongYu
LastEditTime: 2021-02-20 18:55:57
'''
from numpy import hstack
from numpy import array
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score as AUC
from sklearn.model_selection import RandomizedSearchCV
from imblearn.over_sampling import SMOTE
from sklearn.metrics import precision_recall_curve
from imblearn.pipeline import make_pipeline as imbalanced_make_pipeline
import joblib
import os

# 模型存放目录定义
model_root = '/mnt/mydisk/model/TipDMCup20/'

# 基分类器和次级分类器定义，都是二分类，且定义网格超参数搜索
n_estimator = 2
model1 = DecisionTreeClassifier()
# rand_model1 = RandomizedSearchCV(model1, )
model2 = KNeighborsClassifier()
meta_model = LogisticRegression(solver='liblinear')


def train_model(X, y):
    # K折交叉验证
    kfold = KFold(n_splits=2, shuffle=True)

    # 定义out of fold的预测值及其标签

    oof_pred = np.zeros((0, n_estimator))
    oof_y = np.zeros((0, 1))

    #  用基分类器构建oof_pred，也就是次级分类器的特征
    idx = 0
    for train_idx, test_idx in kfold.split(X):
        print("第 %d 折交叉验证" % idx)
        # 获取all fold数据和out of fold数据
        train_X, test_X = X[train_idx], X[test_idx]
        train_y, test_y = y[train_idx], y[test_idx]

        # 用基学习器1进行训练和预测
        pipeline1 = imbalanced_make_pipeline(
            SMOTE(sampling_strategy='minority'),
            model1
        )
        pipeline1.fit(train_X, train_y)
        # model1.fit(train_X, train_y)

        # 二分类，都取标签1的概率
        yhat1 = model1.predict_proba(test_X)[:, 0]
        print("********* 学习器1训练结束 **********")

        # 用基学习器2进行训练和预测
        pipeline2 = imbalanced_make_pipeline(
            SMOTE(sampling_strategy='minority'),
            model2
        )   
        pipeline2.fit(train_X, train_y)
        yhat2 = model2.predict_proba(test_X)[:, 0]
        print("********* 学习器2训练结束 **********")

        # 将其输出组成新的特征
        pred = np.concatenate((yhat1.reshape(-1, 1), yhat2.reshape(-1, 1)), axis=1)
        oof_pred = np.concatenate((oof_pred, pred), axis=0)
        oof_y = np.concatenate((oof_y, test_y.reshape(-1, 1)), axis=0)

        idx += 1
    # 用所有数据训练基础分类器
    pipeline1.fit(X, y)
    pipeline2.fit(X, y)
    
    joblib.dump(model1, os.path.join(model_root, 'model1.json'))
    joblib.dump(model2, os.path.join(model_root, 'model2.json'))

    # 用oof_pred和oof_y训练次级分类器
    pipeline3 = imbalanced_make_pipeline(
        SMOTE(sampling_strategy='minority'),
        meta_model
    )
    pipline3.fit(oof_pred, oof_y)
    joblib.dump(meta_model, os.path.join(model_root, 'meta_model.json'))


def stack_prediction(model1, model2, meta_model, X):
    # make predictions
    yhat1 = model1.predict_proba(X)[:, 0].reshape(-1, 1)
    yhat2 = model2.predict_proba(X)[:, 0].reshape(-1, 1)

    # create input dataset
    meta_X = np.concatenate((yhat1, yhat2), axis=1)

    # predict
    return meta_model.predict(meta_X)


# 对基分类器和次级分类器进行评估
def evaluate_model(X_test, y_test):

    num_pos = np.sum(y_test, axis=0)
    num_neg = y_test.shape[0] - num_pos
    print("正例有：%d, 反例有：%d" % (num_pos, num_neg))


    #  基分类器和次级分类器定义
    model1 = joblib.load(os.path.join(model_root, 'model1.json'))
    model2 = joblib.load(os.path.join(model_root, 'model2.json'))
    meta_model = joblib.load(os.path.join(model_root, 'meta_model.json'))

    #  对初级分类器进行准确率评估
    acc1 = accuracy_score(y_test, model1.predict(X_test))
    acc2 = accuracy_score(y_test, model2.predict(X_test))

    # 对初级分类器进行召回率评估
    recall1 = recall_score(y_test, model1.predict(X_test))
    recall2 = recall_score(y_test, model2.predict(X_test))

    # 对初级分类器进行精准率评估
    preci1 = precision_score(y_test, model1.predict(X_test))
    preci2 = precision_score(y_test, model2.predict(X_test))

    print('Model1 Accuracy: %.3f, Model2 Accuracy: %.3f' % (acc1, acc2))
    print('Model1 Recall: %.3f, Model2 Recall: %.3f' % (recall1, recall2))
    print('Model1 Precision1: %.3f, Model2 Precision: %.3f' % (preci1, preci2))

    #  对次级分类器进行评估
    yhat = stack_prediction(model1, model2, meta_model, X_test)
    acc = accuracy_score(y_test, yhat)
    print('Meta Model Accuracy: %.3f' % (acc))
