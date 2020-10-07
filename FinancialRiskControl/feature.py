import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from tqdm import tqdm
from scipy.stats import pearsonr
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, VarianceThreshold
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import warnings
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, log_loss
warnings.filterwarnings('ignore')

data_train =pd.read_csv('train.csv')
data_test_a = pd.read_csv('testA.csv')

numerical_fea = list(data_train.select_dtypes(exclude = ['object']).columns)
nonumerical_fea = list(filter(lambda x: x not in numerical_fea, list(data_train.columns)))
label = 'isDefault'
numerical_fea.remove(label)
category_fea = list(filter(lambda x: x not in numerical_fea, list(data_train.columns)))
#aveage and most
def fillValue():
    data_train[numerical_fea] = data_train[numerical_fea].fillna(data_train[numerical_fea].mode())
    data_test_a[numerical_fea] = data_test_a[numerical_fea].fillna(data_train[numerical_fea].median())
    #按照众数填充类别型特征
    data_train[category_fea] = data_train[category_fea].fillna(data_train[category_fea].mode())
    data_test_a[category_fea] = data_test_a[category_fea].fillna(data_train[category_fea].mode())
    print(data_train.isnull().sum())
    #把所有缺失值替换为指定的值0
    #data_train = data_train.fillna(0)

    #向用缺失值上面的值替换缺失值
    #data_train = data_train.fillna(axis=0,method='ffill')

    #纵向用缺失值下面的值替换缺失值,且设置最多只填充两个连续的缺失值
    #data_train = data_train.fillna(axis=0,method='bfill',limit=2)


#时间格式处理
def timeFormat():
    for data in [data_train, data_test_a]:
        data['issueDate'] = pd.to_datetime(data['issueDate'],format='%Y-%m-%d')
        startdate = datetime.datetime.strptime('2007-06-01', '%Y-%m-%d')
        #构造时间特征
        data['issueDateDT'] = data['issueDate'].apply(lambda x: x-startdate).dt.days
    print(data_train['employmentLength'].value_counts(dropna=False).sort_index())

#对象类型特征转换到数值
def employmentLength_to_int(s):
    if pd.isnull(s):
        return s
    else:
        return np.int8(s.split()[0])

def objTransferToVal():        
    for data in [data_train, data_test_a]:
        data['employmentLength'].replace(to_replace='10+ years', value='10 years', inplace=True)
        data['employmentLength'].replace('< 1 year', '0 years', inplace=True)
        data['employmentLength'] = data['employmentLength'].apply(employmentLength_to_int)
    print(data['employmentLength'].value_counts(dropna=False).sort_index())

#earliesCreditLine预处理
def eCLPre():
    for data in [data_train, data_test_a]:
        data['earliesCreditLine'] = data['earliesCreditLine'].apply(lambda s: int(s[-4:]))

#部分类别特征
def partOfFea():
    cate_features = ['grade', 'subGrade', 'employmentTitle', 'homeOwnership', 'verificationStatus', 'purpose', 'postCode', 'regionCode', \
                    'applicationType', 'initialListStatus', 'title', 'policyCode']
    for f in cate_features:
        print(f, '类型数：', data_test_a[f].nunique())



#get_dummies() one hot code 
#均方差：如果一个数据分布近似正态，那么大约 68% 的数据值会在均值的一个标准差范围内，
# 大约 95% 会在两个标准差范围内，大约 99.7% 会在三个标准差范围内。
def find_outliers_by_3segama(data, fea):
    data_std = np.std(data[fea])
    data_mean = np.mean(data[fea])
    outliers_cut_off = data_std * 3
    lower_rule = data_mean - outliers_cut_off
    upper_rule = data_mean + outliers_cut_off
    data[fea + '_outliers'] = data[fea].apply(lambda x:str('Exception value') if\
        x > upper_rule or x < lower_rule else 'normal value')
    return data

def showRE():
    #data_train = data_train.copy()
    for fea in numerical_fea:
        data = find_outliers_by_3segama(data_train,fea)
        print(data[fea+'_outliers'].value_counts())
        print(data.groupby(fea+'_outliers')['isDefault'].sum())
        print('*'*10)

#删除异常值
def delExp():
    for fea in numerical_fea:
        data_train = data_train[data_train[fea+'_outliers']=='normal value']
        data_train = data_train.reset_index(drop=True) 

#数据分箱
def dataDis():
    #固定宽度分箱
    data = find_outliers_by_3segama(data_train,numerical_fea)
    # 通过除法映射到间隔均匀的分箱中，每个分箱的取值范围都是loanAmnt/1000
    data['loanAmnt_bin1'] = np.floor_divide(data['loanAmnt'], 1000)
    ## 通过对数函数映射到指数宽度分箱
    data['loanAmnt_bin2'] = np.floor(np.log10(data['loanAmnt']))
    #分位数分箱
    data['loanAmnt_bin3'] = pd.qcut(data['loanAmnt'], 10, labels=False)

def featureSelection():
    print('Feature Selection\n')
    #Variance selection
    #VarianceThreshold(threshold=3).fit_transform(train, target_train)

    #Pearson correlation
    #SelectKBest(k=5).fit_transform(train, target_train)

    #Ka fang 
    #SelectKBesst(chi2, k=5).fit_transform(train,target_train)

    #


#python feature.py

showRE()  

