'''
Descripttion: Stacking
Version: 1.0
Author: ZhangHongYu
Date: 2021-02-18 13:11:38
LastEditors: ZhangHongYu
LastEditTime: 2021-02-21 17:40:03
'''
from numpy import array
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.model_selection import RandomizedSearchCV
from imblearn.over_sampling import SMOTE
from sklearn.metrics import precision_recall_curve
from sklearn import model_selection
import joblib
import os

# 模型存放目录定义
model_root = '/home/macong/project/model'
k = 2  # 交叉验证折数

#  基分类器和次级分类器定义，都是二分类，且定义网格超参数搜索
#  基分类器定义
models={}

models.update({'dt': DecisionTreeClassifier(random_state=0)})
models.update({'rf': 
    RandomForestClassifier(random_state=0)
})
models.update({'knn': KNeighborsClassifier()})
models.update({'et': 
    ExtraTreesClassifier(random_state=0)
})
# models.update({'mlp': 
#     MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15,), random_state=1)
# })


# 基分类器超参数搜索范围定义
param_grids = {}
param_grids.update({
    'dt':
    { 'min_samples_split': [2, 4], 'max_depth': [4]}
})
param_grids.update({
    'rf':
    {'n_estimators': [500], 'min_samples_split': [2, 3], 'max_depth': [4]}
})
param_grids.update({
    'knn':
    {'n_neighbors':[ 10] }
})
param_grids.update({
    'et':
   {'n_estimators': [500], 'min_samples_split': [3, 4], 'max_depth': [4]}
})

# param_grids.update({
#     'mlp':{}
# })


 #  完成超参数网格搜索后的模型
model_grids={}
for name,  param in param_grids.items():
    model_grids[name] = model_selection.RandomizedSearchCV(models[name], param, n_jobs=-1, cv=10, verbose=1,scoring='roc_auc')
    # model_grids[name] = models[name]

# 次级分类器定义
# meta_model =  XGBClassifier(n_estimators=2000, max_depth=4, min_child_weight=2, gamma=0.9, subsample=0.8,  colsample_bytree=0.8, objective='binary:logistic', nthread=-1, scale_pos_weight=1)
meta_model = LogisticRegression(solver='liblinear')
def train_model(X, y):
    # SMOTE过采样
    smo = SMOTE(random_state=42, n_jobs=-1 )
    X_sampling,  y_sampling = smo.fit_resample(X, y)

    # K折交叉验证
    kfold = KFold(n_splits=k, shuffle=True)

    # 定义out of fold的预测值及其标签

    oof_pred = np.zeros((0, len(models)))
    oof_y = np.zeros((0, 1))

    print("********************  training  ***********************")
    idx = 0
    #  用基分类器构建oof_pred，也就是次级分类器的特征
    for train_idx, test_idx in kfold.split(X_sampling):
        print("第 %d 折交叉验证" % idx)
        # 获取all fold数据和out of fold数据
        train_X, test_X = X_sampling[train_idx], X_sampling[test_idx]
        train_y, test_y = y_sampling[train_idx], y_sampling[test_idx]

        y_hats = []
        # 依次用基学习器进行训练和预测
        for name,  model_grid  in model_grids.items():
            # 注意，这里没有实际修改model_grid[name]，
            # 此处是为了得到oof_pred,后面才会用全部数据集训练,
            # 这里每折训练所得的参数不会保存
            model_grid.fit(train_X, train_y)
            # y_hats.append(model_grid.predict(test_X).reshape(-1, 1))
            y_hats.append(model_grid.predict(test_X).reshape(-1, 1))
            # 评估k折验证过程中初级分类器在训练集上的表现
            acc = accuracy_score(train_y, model_grid.predict(train_X))
            recall = recall_score(train_y, model_grid.predict(train_X))
            preci = precision_score(train_y, model_grid.predict(train_X))
            auc =roc_auc_score(train_y, model_grid.predict(train_X))
            print(" Sub model %s accuracy: : %.3f " % (name, acc))
            print(" Sub model %s recalll: %.3f " % (name, recall))
            print(" Sub model %s precision: %.3f " % (name, preci))
            print(" Sub model %s auc: %.3f " % (name, auc))

        # 将其输出组成新的特征
        pred = np.concatenate(
            (tuple(y_hats)), axis=1)
        oof_pred = np.concatenate((oof_pred, pred), axis=0)
        oof_y = np.concatenate((oof_y, test_y.reshape(-1, 1)), axis=0)
        idx += 1

    # 用所有数据训练基础分类器
    for name, _  in model_grids.items():
        # 这里才对model_grids[name]进行实际修改
        model_grids[name].fit(X_sampling, y_sampling)
        joblib.dump(model_grid, os.path.join(model_root, name +'.json'))

    # 用所有数据训练次级分类器
    meta_model. fit(oof_pred,oof_y)

    # 评估次级分类器在训练集上的误差
    acc = accuracy_score(oof_y, meta_model.predict(oof_pred))
    recall = recall_score(oof_y, meta_model.predict(oof_pred))
    preci = precision_score(oof_y, meta_model.predict(oof_pred))
    auc = roc_auc_score(oof_y, meta_model.predict(oof_pred))
    print(" Meta model  accuracy: : %.3f "  % acc)
    print(" Meta model  recalll: %.3f " % recall)
    print(" Meta model  precision: %.3f " % preci)
    print(" Meta model  auc: %.3f " % auc)
    joblib.dump(meta_model, os.path.join(model_root, 'meta_model.json'))


def stack_prediction(model_grids,  meta_model, X):
    # make predictions
    y_hats = []
    for _, model_grid in model_grids.items():
        y_hats.append(model_grid.predict(X).reshape(-1, 1))

    # create input dataset
    meta_X = np.concatenate(
        tuple(y_hats), axis=1)

    # predict
    return meta_model.predict(meta_X)


# 对基分类器和次级分类器进行评估
def evaluate_model(X_test, y_test):

    print("********************  evaluation  ***********************")

    #  基分类器和次级分类器定义
    for name, _ in model_grids.items():
        model_grids[name] = joblib.load(os.path.join(model_root, name+'.json')) 
        # 对初级分类器进行评估
        model_grid = model_grids[name]
        acc = accuracy_score(y_test, model_grid.predict(X_test))
        recall = recall_score(y_test, model_grid.predict(X_test))
        preci = precision_score(y_test, model_grid.predict(X_test))
        auc = roc_auc_score(y_test, model_grid.predict(X_test))
        print(" Sub model %s accuracy: : %.3f " % (name, acc))
        print(" Sub model %s recalll: %.3f " % (name, recall))
        print(" Sub model %s precision: %.3f " % (name, preci))
        print(" Sub model %s auc: %.3f " % (name, auc))


    meta_model = joblib.load(os.path.join(model_root, 'meta_model.json'))
    #  对次级分类器进行评估
    y_hat = stack_prediction(model_grids,  meta_model, X_test)
    acc = accuracy_score(y_test,  y_hat)
    recall = recall_score(y_test, y_hat)
    preci = precision_score(y_test, y_hat)
    auc = roc_auc_score(y_test, y_hat)
    print('Meta model accuracy: %.3f' % (acc))
    print('Meta model recall: %.3f' % (recall))
    print('Meta model precision %.3f' % (preci))
    print('Meta model auc  %.3f' % (auc))

