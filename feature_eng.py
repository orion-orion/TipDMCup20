'''
Descripttion: 
Version: 1.0
Author: ZhangHongYu
Date: 2021-02-18 13:15:08
LastEditors: ZhangHongYu
LastEditTime: 2021-02-24 10:49:17
'''
import pandas as pd
import numpy as np
import os
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.model_selection import RandomizedSearchCV
from imblearn.over_sampling import SMOTE
from sklearn import model_selection
from sklearn.feature_selection import SelectFromModel
from sklearn import decomposition
import joblib

# 数据存放目录定义
data_root = '/public1/home/sc80074/TipDMCup20/data/A题全部数据/'
# 用于特征选择的模型的目录
features_model_root = '/public1/home/sc80074/TipDMCup20/features_model'
# 用于保存模型特征信息的目录
features_imp_root = '/public1/home/sc80074/TipDMCup20/features_imp'

top_n = 30 #选出的top-n特征

pca_dim = 10 # pca 降维后的维度

# 用于特征选择的模型定义
models={}


models.update({'dt':
    DecisionTreeClassifier(random_state=0)
})
models.update({'rf': 
    RandomForestClassifier(random_state=0)
})
models.update({'et': 
    ExtraTreesClassifier(random_state=0)
})
models.update({'xgb': 
    XGBClassifier(random_state=0)
})
# models.update({'mlp': 
#     MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15,), random_state=1)
# })


# 用于特征选择的模型的超参数搜索范围定义
param_grids = {}
param_grids.update({
    'dt':
    { 'min_samples_split': [2, 4], 'max_depth': [12]}
})
param_grids.update({
    'rf':
    {'n_estimators': [500], 'min_samples_split': [2, 3], 'max_depth': [12],'n_jobs':[-1]}
})
param_grids.update({
    'et':
   {'n_estimators': [500], 'min_samples_split': [3, 4], 'max_depth': [12],'n_jobs':[-1]}
})
param_grids.update({
    'xgb':
   {'n_estimators': [500],  'max_depth': [2], 'objective':['binary:logistic'], 'eval_metric':['logloss'],'use_label_encoder':[False],'nthread':[-1]}
})

# param_grids.update({
#     'mlp':
#     {'solver':['lbfgs'], 'alpha':[1e-5], 'hidden_layer_sizes':[(15,)], 'random_state':[1] }
# })


 #  完成超参数网格搜索后的模型
model_grids={}
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
for name,  param in param_grids.items():
    model_grids[name] = model_selection.GridSearchCV(models[name], param, n_jobs=-1, cv=kfold, verbose=1,scoring='f1')
    # model_grids[name] = models[name]


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


#用模型对特征进行选择
def feature_selection(X, y):
    # SMOTE过采样
    smo = SMOTE(random_state=42, n_jobs=-1 )
    X_sampling,  y_sampling = smo.fit_resample(X, y)

     # 用所有数据训练用于特征选择的模型
    # for name, _  in model_grids.items():
    #         # 这里才对model_grids[name]进行实际修改
    #         model_grids[name].fit(X_sampling, y_sampling)
    #         joblib.dump(model_grids[name], os.path.join(features_model_root, name +'.json'))
    #         print(" features selection model %s has been trained " % (name))

    # 加载用于特征选择的模型并选出top-n的特征
    features_top_n_list = []
    for name, _ in model_grids.items():
        model_grids[name] = joblib.load(os.path.join(features_model_root, name+'.json')) 
        model_grid = model_grids[name]
        features_imp_sorted = pd.DataFrame({'feature': list(X),
                                            'importance': model_grid.best_estimator_.feature_importances_}).sort_values('importance', ascending=False)
        features_top_n = features_imp_sorted.head(top_n)['feature']
        features_top_n_imp =  features_imp_sorted.head(top_n)['importance']
        features_top_n_list.append(features_top_n)
        features_output = pd.DataFrame({'features_top_n':features_top_n, 'importance':features_top_n_imp})
        features_output.to_csv(os.path.join(features_imp_root, name+'_features_top_n_importance.csv'))
    features_top_n = pd.concat(features_top_n_list, ignore_index=True).drop_duplicates()
    X = pd.DataFrame(X[features_top_n])
    return X


def data_preprocess(data):
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

    # 字符独热编码，数值归一化
    for col in data.columns:
        if col == '是否高转送':  # 跳过标签列
            continue
        if str(data[col].dtype) == 'object':
            # 字符->数值
            data.loc[:, col] = pd.factorize(
                data[col])[0]
            # 获取dummy编码
            dummies_df = pd.get_dummies(data[col], prefix=str(col))
            data = data.drop(col, axis=1)
            data = data.join(dummies_df)
        else:
            # 对数值特征归一化
            scaler = preprocessing.StandardScaler().fit(
                np.array(data[col]).reshape(-1, 1))
            data.loc[:,  col] = scaler.transform(np.array(data[col]).reshape(-1, 1))        
    return data

def data_decomposition(X):
    pca = decomposition.PCA()
    pca.fit(X)
    pca.n_components = pca_dim
    return pca.fit_transform(X)