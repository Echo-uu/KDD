import pandas as pd
import os
import gc
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.linear_model import SGDRegressor, LinearRegression, Ridge
from sklearn.preprocessing import MinMaxScaler
import math
import numpy as np
from tqdm import tqdm
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, log_loss, confusion_matrix, precision_recall_curve, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import time
import warnings
warnings.filterwarnings('ignore')


train = pd.read_csv('train.csv')
testA = pd.read_csv('testA.csv')

print('Train data shape:',train.shape)
print('TestA data shape:',testA.shape)

print(train.head())

data = pd.concat([train, testA], axis = 0, ignore_index = True)

#Confuse Matrix
y_pred = [0, 1, 0, 1]
y_true = [0, 1, 1, 0]
print(confusion_matrix(y_true, y_pred))

#Accuracy
print('ACC: ', accuracy_score(y_true, y_pred))

#precision, recall, F1-score
print('Precision: ', metrics.precision_score(y_true, y_pred))
print('Recall: ', metrics.recall_score(y_true, y_pred))
print('F1-score: ', metrics.f1_score(y_true, y_pred))

#P-R曲线
y_pred = [0.1, 0.4, 0.35, 0.8, 0.19, 0.48, 0.95, 0.81, 0.44, 0.99]
y_true = [0, 1, 1, 0, 1, 0, 1, 1, 0, 1]
precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
print('thresholds: ', thresholds)
plt.plot(precision, recall)
plt.show()

#python baseline.py

#ROC曲线
FPR, TPR, thresholds = roc_curve(y_true, y_pred)
print('\nthresholds: ', thresholds)
plt.title('ROC')
plt.plot(FPR, TPR, 'b')
plt.plot([0, 1], [0, 1], 'r--')
plt.ylabel('TPR')
plt.xlabel('FPR')
plt.show()

#AUC
y_true = np.array([0, 0, 1, 1])
y_scores = np.array([0.1, 0.4, 0.35, 0.8])
print('AUC socre:',roc_auc_score(y_true, y_scores))

## KS值 在实际操作时往往使用ROC曲线配合求出KS值
from sklearn.metrics import roc_curve
y_pred = [0, 1, 1, 0, 1, 1, 0, 1, 1, 1]
y_true = [0, 1, 1, 0, 1, 0, 1, 1, 1, 1]
FPR,TPR,thresholds=roc_curve(y_true, y_pred)
KS=abs(FPR-TPR).max()
print('KS值：',KS)

#评分卡 不是标准评分卡
def Score(prob,P0=600,PDO=20,badrate=None,goodrate=None):
    P0 = P0
    PDO = PDO
    theta0 = badrate/goodrate
    B = PDO/np.log(2)
    A = P0 + B*np.log(2*theta0)
    score = A-B*np.log(prob/(1-prob))
    return score