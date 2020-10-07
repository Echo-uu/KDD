import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import csv
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
lgb_train = pd.read_csv('train answer.csv')
#numerical_fea = list(data_train.select_dtypes(exclude = ['object']).columns)
#nonumerical_fea = list(filter(lambda x: x not in numerical_fea, list(data_train.columns)))
#label = 'isDefault'
#numerical_fea.remove(label)
#category_fea = list(filter(lambda x: x not in numerical_fea, list(data_train.columns)))

for data in [data_train, data_test_a]:
    data.drop(['issueDate','id'], axis=1,inplace=True)

data_train = data_train.fillna(axis=0,method='ffill')
#x_train = data_train.drop(['isDefault','id'], axis=1)

for col in tqdm(['subGrade', 'grade', 'employmentLength', 'earliesCreditLine']):
    le = LabelEncoder()
    le.fit(list(data_train[col].astype(str).values) + list(data_test_a[col].astype(str).values))
    data_train[col] = le.transform(list(data_train[col].astype(str).values))
    data_test_a[col] = le.transform(list(data_test_a[col].astype(str).values))
print('Label Encoding 完成')

features = [f for f in data_train.columns if f not in ['id','issueDate','isDefault'] and '_outliers' not in f]
x_train = data_train[features]
x_test = data_test_a[features]
y_train = data_train['isDefault']
lgb_train = lgb_train['pred']


#print(x_train.info())
#print(x_test.info())

print(y_train.head())
print(lgb_train.head())
print(roc_auc_score(y_train, lgb_train))


