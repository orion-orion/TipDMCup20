'''
Descripttion: 
Version: 1.0
Author: ZhangHongYu
Date: 2021-02-18 13:15:08
LastEditors: ZhangHongYu
LastEditTime: 2022-03-24 15:19:46
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
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import VarianceThreshold
from scipy.interpolate import interp1d
import joblib

# 数据存放目录定义
data_root = './data'
# 用于特征选择的模型的目录
features_model_root = './features_model'
# 用于保存模型特征信息的目录
features_imp_root = './features_imp'

pca_dim = 10 # pca 降维后的维度

top_n = 30

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
    data1 = pd.read_csv(os.path.join(data_root, '基础数据.csv'), encoding='GB2312')
    data2 = pd.read_csv(os.path.join(data_root, '年数据.csv'), encoding='GB2312')
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

    #后面join是默认按照index来的，删除缺失值后重新设置index
    combined_data = combined_data.reset_index(drop=True) 

    labels = combined_data['是否高转送'].to_list()
    # 我们根据上一年的特征预测下一年是否高送转，故标签是下一年的
    for i in range(len(labels)-1):
        labels[i] = labels[i+1]
    combined_data['是否高转送'] = pd.Series(labels)
    return combined_data


#用模型对特征进行选择
def feature_selection(X, y, mod):
    #根据阈值移除低方差特征
    # 假设是布尔特征，我们想要移除特征值为0或者为1的比例超过0.8的特征
    # 布尔特征为Bernoulli随机变量，方差为p(1-p)
    # 该方法的输出会把特征名去掉，故不采用
    # sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
    # X_sel = sel.fit_transform(X)
    features_top_n_list = []
    if mod == 'retrain': #如果是'ratrain'则重新对特征选择的模型进行训练，并保存结果
        if not os.path.exists(features_model_root):
            os.makedirs(features_model_root)
        # SMOTE过采样
        smo = SMOTE(random_state=42, n_jobs=-1 )
        X_sampling,  y_sampling = smo.fit_resample(X, y)

        #用所有数据训练用于特征选择的模型
        for name, _  in model_grids.items():
                # 这里才对model_grids[name]进行实际修改
                model_grids[name].fit(X_sampling, y_sampling)
                joblib.dump(model_grids[name], os.path.join(features_model_root, name +'.json'))
                print(" features selection model %s has been trained " % (name))

        if not os.path.exists(features_imp_root):
            os.makedirs(features_imp_root)
        
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

    elif mod == 'load': #如果是则直接加载已经得到的特征选择结果
        if not os.path.exists(features_imp_root):
            raise IOError("cant find the features imp directory: %s" % features_imp_root)
        for name, _ in model_grids.items():
            features_top_n = pd.read_csv(os.path.join(features_imp_root, name+'_features_top_n_importance.csv'))['features_top_n']
            features_top_n_list.append(features_top_n)

    else:
        raise IOError("invalid mod!") 
        
        

    # 加载用于特征选择的模型并选出top-n的特征
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

    # 对缺失值进行填补
    for feature in features_fillna:
        # 如果是非数值型特征或者是整型离散数值，用众数填补
        #将列按出现频率由高到低排序，众数即第一行，inplace表示原地修改
        if str(data[feature].dtype) == 'object' or str(data[feature].dtype) =='int64':
            data.loc[:,  feature] = data[feature].fillna(
                data[feature].mode().iloc[0]
            )
        #浮点连续数值型特征插值填补+平均数处理边缘
        else:
            #先将中间的数据插值处理
            data.loc[:,  feature] = data[feature].interpolate( method="zero", axis=0, limit_direction='both')
            #边缘直接填充平均数
            data.loc[:,  feature] = data[feature].fillna(
                data[feature].mean()
            )
            if np.isnan(data.loc[:, feature]).any():
                print(data.loc[:, feature])
        #   print(data[feature])

    # 字符独热编码与数值归一化
    # 先处理所属概念板块这一列
    all_types = {} #总共的types种类
    for idx, combined_type in data['所属概念板块'].items():
        types = combined_type.split(';')
        dict_type = {}
        for type_ in types:
            dict_type[type_] = 1
            all_types[type_] = 1
        data['所属概念板块'][idx] = dict_type
    for idx, dict_type in data['所属概念板块'].items():
        for k in all_types.keys():
            if k in dict_type.keys():
                continue
            else:
                data['所属概念板块'][idx][k] = 0
    for col in data.columns:
        if col == '是否高转送':  # 跳过标签列
            continue
        if col == '股票编号':
            #  这里标称形不是连续的，不能直接转换为数值
            # data.loc[:, col] = pd.factorize(
            #     data[col])[0]
            # 只能转换为dummy编码，以下为获取dummy编码, 后面还要用，暂时保存副本
            dummies_df = pd.get_dummies(data[col], prefix=str(col))
            data = data.join(dummies_df)
            continue
        if col == '所属概念板块': #对所属概念板块单独处理
            vec = DictVectorizer()
            arr = np.array(vec.fit_transform(data[col].to_list()).toarray())
            data = data.drop(col, axis=1)
            for i in range(arr.shape[1]):
                data = data.join(pd.DataFrame({(col+str(i)): arr[:, i]}))
            continue
        if str(data[col].dtype) == 'object':
            #  这里标称形不是连续的，不能直接转换为数值
            # data.loc[:, col] = pd.factorize(
            #     data[col])[0]
            # 只能转换为dummy编码，以下为获取dummy编码
            dummies_df = pd.get_dummies(data[col], prefix=str(col))
            data = data.drop(col, axis=1)
            data = data.join(dummies_df)
        else:
            # 对数值特征z-score标准化
            scaler = preprocessing.StandardScaler().fit(
                np.array(data[col]).reshape(-1, 1))
                #年份特征转换后要保留副本后面划分数据集用
            result = scaler.transform(np.array(data[col]).reshape(-1, 1))  
            if col == '年份（年末）': #年份特征要保留原来副本后面划分样本用
                copy = data[col].to_list()
                data.loc[:, col] = result
                data = data.join(pd.DataFrame({'年份copy':copy}))
            else:
                data.loc[:, col] = result #其他特征直接覆盖即可
            # 对数值特征二范数归一化，该操作独立对待样本，无需对normalizer进行fit
            # 但dummy编码不好处理，故不考虑之
            # data.loc[:, col] = preprocessing.normalize(np.array(data[col]).reshape(-1, 1),norm='l2')

    return data

def data_decomposition(X):
    pca = decomposition.PCA()
    pca.fit(X)
    pca.n_components = pca_dim
    return pca.fit_transform(X)