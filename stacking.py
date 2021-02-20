'''
Descripttion: Stacking
Version: 1.0
Author: ZhangHongYu
Date: 2021-02-18 13:11:38
LastEditors: ZhangHongYu
LastEditTime: 2021-02-20 21:57:58
'''
from numpy import hstack
from numpy import array
import numpy as np
from sklearn.datasets import make_blobs
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
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score as AUC
from sklearn.model_selection import RandomizedSearchCV
from imblearn.over_sampling import SMOTE
from sklearn.metrics import precision_recall_curve
from sklearn import model_selection
from imblearn.pipeline import make_pipeline as imbalanced_make_pipeline
import joblib
import os

# 模型存放目录定义
model_root = '/home/macong/project/model'

# 基分类器和次级分类器定义，都是二分类，且定义网格超参数搜索
n_estimator = 5

# dt_param_grid = {'min_samples_split': [2, 4], 'max_depth': [20]}
# model1 = DecisionTree(random_state=0)

# rf_param_grid = {'n_estimators': [500], 'min_samples_split': [2, 3], 'max_depth': [20]}
# model2 = RandomForestClassifier(random_state=0)

# ada_param_grid = {'n_estimators': [500], 'learning_rate': [0.01, 0.1]}
# model3 = AdaBoostClassifier(random_state=0)

# et_param_grid = {'n_estimators': [500], 'min_samples_split': [3, 4], 'max_depth': [20]}
# model4 = ExtraTreesClassifier(random_state=0)

# gb_param_grid = {'n_estimators': [500], 'learning_rate': [0.01, 0.1], 'max_depth': [20]}
# model5 = GradientBoostingClassifier(random_state=0)

# meta_model =  XGBClassifier(n_estimators=2000, max_depth=4, min_child_weight=2, gamma=0.9, subsample=0.8,
#                         colsample_bytree=0.8, objective='binary:logistic', nthread=-1, scale_pos_weight=1)



model1 = DecisionTreeClassifier(max_depth=8)
model2 = RandomForestClassifier(n_estimators=500, warm_start=True, max_features='sqrt', max_depth=6,
                                min_samples_split=3, min_samples_leaf=2, n_jobs=-1, verbose=0)
model3 = AdaBoostClassifier(n_estimators=500, learning_rate=0.1)
model4 = ExtraTreesClassifier(n_estimators=500, n_jobs=-1, max_depth=8, min_samples_leaf=2, verbose=0)
model5 = GradientBoostingClassifier(n_estimators=500, learning_rate=0.008,
                                    min_samples_split=3, min_samples_leaf=2, max_depth=5, verbose=0)
meta_model =  XGBClassifier(n_estimators=2000, max_depth=4, min_child_weight=2, gamma=0.9, subsample=0.8,
                        colsample_bytree=0.8, objective='binary:logistic', nthread=-1, scale_pos_weight=1)


def train_model(X, y):
    # SMOTE过采样
    smo = SMOTE(random_state=42, n_jobs=-1 )
    X_sampling,  y_sampling = smo.fit_resample(X, y)

    # K折交叉验证
    kfold = KFold(n_splits=5, shuffle=True)

    # 定义out of fold的预测值及其标签

    oof_pred = np.zeros((0, n_estimator))
    oof_y = np.zeros((0, 1))


    # 网格搜索超参数
    # dt_grid = model_selection.GridSearchCV(et_est, et_param_grid, n_jobs=-1, cv=10, verbose=1)
    # rf_grid = model_selection.GridSearchCV(et_est, et_param_grid, n_jobs=-1, cv=10, verbose=1)
    # ada_grid = model_selection.GridSearchCV(et_est, et_param_grid, n_jobs=-1, cv=10, verbose=1)
    # et_grid = model_selection.GridSearchCV(et_est, et_param_grid, n_jobs=-1, cv=10, verbose=1)
    # gb_grid = model_selection.GridSearchCV(et_est, et_param_grid, n_jobs=-1, cv=10, verbose=1)

    idx = 0
    #  用基分类器构建oof_pred，也就是次级分类器的特征
    for train_idx, test_idx in kfold.split(X_sampling):
        print("第 %d 折交叉验证" % idx)
        # 获取all fold数据和out of fold数据
        train_X, test_X = X_sampling[train_idx], X_sampling[test_idx]
        train_y, test_y = y_sampling[train_idx], y_sampling[test_idx]

        # 用基学习器1进行训练和预测
        model1.fit(train_X, train_y)
        # 二分类，都取标签1的概率
        yhat1 = model1.predict_proba(test_X)[:, 0]
        print("********* 学习器1训练结束 **********")

        # 用基学习器2进行训练和预测
        model2.fit(train_X, train_y)
        yhat2 = model2.predict_proba(test_X)[:, 0]
        print("********* 学习器2训练结束 **********")

        # 用基学习器3进行训练和预测
        model3.fit(train_X, train_y)
        yhat3 = model3.predict_proba(test_X)[:, 0]
        print("********* 学习器3训练结束 **********")

        # 用基学习器4进行训练和预测
        model4.fit(train_X, train_y)
        yhat4 = model4.predict_proba(test_X)[:, 0]
        print("********* 学习器4训练结束 **********")


        # 用基学习器5进行训练和预测
        model5.fit(train_X, train_y)
        yhat5 = model5.predict_proba(test_X)[:, 0]
        print("********* 学习器5训练结束 **********")

        # 将其输出组成新的特征
        pred = np.concatenate(
            (yhat1.reshape(-1, 1), 
            yhat2.reshape(-1, 1),
            yhat3.reshape(-1, 1),
            yhat4.reshape(-1, 1),
            yhat5.reshape(-1, 1)), axis=1)
        oof_pred = np.concatenate((oof_pred, pred), axis=0)
        oof_y = np.concatenate((oof_y, test_y.reshape(-1, 1)), axis=0)

        idx += 1
    # 用所有数据训练基础分类器
    model1.fit(X_sampling,  y_sampling)
    model2.fit(X_sampling,  y_sampling)
    model3.fit(X_sampling,  y_sampling)
    model4.fit(X_sampling,  y_sampling)
    model5.fit(X_sampling,  y_sampling)

    joblib.dump(model1, os.path.join(model_root, 'model1.json'))
    joblib.dump(model2, os.path.join(model_root, 'model2.json'))
    joblib.dump(model3, os.path.join(model_root, 'model3.json'))
    joblib.dump(model4, os.path.join(model_root, 'model4.json'))
    joblib.dump(model5, os.path.join(model_root, 'model5.json'))


    meta_model. fit(oof_pred,oof_y)
    joblib.dump(meta_model, os.path.join(model_root, 'meta_model.json'))


def stack_prediction(model1, model2, model3, model4, model5, meta_model, X):
    # make predictions
    yhat1 = model1.predict_proba(X)[:, 0].reshape(-1, 1)
    yhat2 = model2.predict_proba(X)[:, 0].reshape(-1, 1)
    yhat3 = model3.predict_proba(X)[:, 0].reshape(-1, 1)
    yhat4 = model4.predict_proba(X)[:, 0].reshape(-1, 1)
    yhat5 = model5.predict_proba(X)[:, 0].reshape(-1, 1)

    # create input dataset
    meta_X = np.concatenate((
        yhat1, yhat2, yhat3,yhat4,yhat5), axis=1)

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
    model3 = joblib.load(os.path.join(model_root, 'model3.json'))
    model4 = joblib.load(os.path.join(model_root, 'model4.json'))
    model5 = joblib.load(os.path.join(model_root, 'model5.json'))
    meta_model = joblib.load(os.path.join(model_root, 'meta_model.json'))

    #  对初级分类器进行准确率评估
    acc1 = accuracy_score(y_test, model1.predict(X_test))
    acc2 = accuracy_score(y_test, model2.predict(X_test))
    acc3 = accuracy_score(y_test, model3.predict(X_test))
    acc4 = accuracy_score(y_test, model4.predict(X_test))
    acc5 = accuracy_score(y_test, model5.predict(X_test))




    # 对初级分类器进行召回率评估
    recall1 = recall_score(y_test, model1.predict(X_test))
    recall2 = recall_score(y_test, model2.predict(X_test))
    recall3 = recall_score(y_test, model3.predict(X_test))
    recall4 = recall_score(y_test, model4.predict(X_test))
    recall5 = recall_score(y_test, model5.predict(X_test))



    # 对初级分类器进行精准率评估
    preci1 = precision_score(y_test, model1.predict(X_test))
    preci2 = precision_score(y_test, model2.predict(X_test))
    preci3 = precision_score(y_test, model3.predict(X_test))
    preci4 = precision_score(y_test, model4.predict(X_test))
    preci5 = precision_score(y_test, model5.predict(X_test))

    print('Model1 Accuracy: %.3f, Model2 Accuracy: %.3f' % (acc1, acc2))
    print('Model1 Recall: %.3f, Model2 Recall: %.3f' % (recall1, recall2))
    print('Model1 Precision1: %.3f, Model2 Precision: %.3f' % (preci1, preci2))

    #  对次级分类器进行评估
    y_hat = stack_prediction(model1, model2, model3, model4, model5,  meta_model, X_test)
    acc = accuracy_score(y_test,  y_hat)
    recall = recall_score(y_test, y_hat)
    preci = precision_score(y_test, y_hat)
    print('Meta Model Accuracy: %.3f' % (acc))
    print('Meta Model Recall: %.3f' % (recall))
    print('Meta Model Precision %.3f' % (preci))
