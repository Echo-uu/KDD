from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score
from matplotlib import pyplot as plt
from xgboost import XGBClassifier as xgb
from lightgbm import LGBMClassifier
from catboost import CatBoostRegressor 
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import warnings
from sklearn.model_selection import StratifiedKFold, KFold
warnings.filterwarnings('ignore')


def rf():
    # generate 2 class dataset by ouselves
    X, y = make_classification(n_samples=2000, n_classes=2, random_state=1)
    # split datasets into train/test sets (80%~20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
    # generate a no model prediction
    nm_probs = [0 for _ in range(len(y_test))]
    # fit a model
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    # predict probabilities
    rf_probs = model.predict(X_test)
    # calculate scores
    nm_auc = roc_auc_score(y_test, nm_probs)
    rf_auc = roc_auc_score(y_test, rf_probs)
    # summarize scores
    print('No model: ROC AUC=%.2f' % (nm_auc))
    print('Random forest: ROC AUC=%.2f' % (rf_auc))
    # calculate roc curves
    nm_fpr, nm_tpr, _ = roc_curve(y_test, nm_probs)
    rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_probs)
    # plot the roc curve for the model
    plt.plot(nm_fpr, nm_tpr, linestyle='--', label='No model AUC = %0.2f'% nm_auc)
    plt.plot(rf_fpr, rf_tpr, marker='.', label='Random forest AUC = %0.2f'% rf_auc)
    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # show the legend
    plt.legend()
    # show the plot
    plt.show()



data_train =pd.read_csv('F:/data/FinancialRiskControl/train.csv')
data_test_a = pd.read_csv('F:/data/FinancialRiskControl/testA.csv') 

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
x_valid = data_test_a[features]
y_train = data_train['isDefault']

trn_x, val_x, trn_y, val_y = train_test_split(x_train, y_train)

clf = xgb()
clf.fit(trn_x, trn_y)
pre = clf.predict(val_x)
print(roc_auc_score(val_y, pre))

lgb = LGBMClassifier()
lgb.fit(trn_x, trn_y)
pre = lgb.predict(val_x)
print(roc_auc_score(val_y, pre))

cat = CatBoostRegressor()
cat.fit(trn_x, trn_y)
pre = cat.predict(val_x)
print(roc_auc_score(val_y, pre))



