import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import pandas.util.testing as tm
import pandas_profiling
import warnings
warnings.filterwarnings('ignore')

data_train = pd.read_csv('train.csv')
data_test_a = pd.read_csv('testA.csv')

data_train_sample = pd.read_csv("train.csv", nrows = 5)

numerical_fea = list(data_train.select_dtypes(exclude = ['object']).columns)
category_fea = list(filter(lambda x: x not in numerical_fea, list(data_train.columns)))

def readFile():
    print(data_train_sample.head())
    #设置chunksize参数，来控制每次迭代数据的大小
    chunker = pd.read_csv('train.csv', chunksize = 5)
    for item in chunker:
        print(type(item))
        #<class 'pandas.core.frame.DataFrame'>
        print(len(item))
        #5
    
def overview():
    print(data_test_a.shape)
    #print(data_train.shape)
    #print(data_train.columns)
    print(data_train.info())
    #总体粗略的查看数据集各个特征的一些基本统计量
    print(data_train.describe())

#查看缺失值
def lossValue():
    print(f'There are {data_train.isnull().any().sum()} columns in train dataset with missing values.')
    have_null_fea_dict = (data_train.isnull().sum()/len(data_train)).to_dict()
    fea_null_moreThanHalf = {}
    for key,value in have_null_fea_dict.items():
        if value > 0.5:
            fea_null_moreThanHalf[key] = value
    print(fea_null_moreThanHalf)
    missing = data_train.isnull().sum()/len(data_train)
    missing = missing[missing > 0]
    missing.sort_values(inplace=True)
    missing.plot.bar()
    plt.show()

def singleValue():
    one_value_fea = [col for col in data_train.columns if data_train[col].nunique() <= 1]
    one_value_fea_test = [col for col in data_test_a.columns if data_test_a[col].nunique() <= 1]
    print(one_value_fea)
    print(one_value_fea_test)
    print(f'There are {len(one_value_fea)} columns in train dataset with one unique value.')
    print(f'There are {len(one_value_fea_test)} columns in test dataset with one unique value.')

#特征数值类型
def specificValue():
    print(numerical_fea)
    print(category_fea)
    print(data_train.grade)

#过滤数值型类别特征
def get_numerical_serial_fea(data, feas):
    numerical_serial_fea = []
    numerical_noserial_fea = []
    for fea in feas:
        temp = data[fea].nunique()
        if temp <= 10:
            numerical_noserial_fea.append(fea)
            continue
        numerical_serial_fea.append(fea)
    return numerical_serial_fea,numerical_noserial_fea

numerical_serial_fea,numerical_noserial_fea = get_numerical_serial_fea(data_train,numerical_fea)

def nonumericalValue():
    print(data_train['term'].value_counts())
    print(data_train['homeOwnership'].value_counts())
    print(data_train['verificationStatus'].value_counts())
    print(data_train['initialListStatus'].value_counts())
    print(data_train['applicationType'].value_counts())#离散型变量
    print(data_train['policyCode'].value_counts())#离散型变量，无用，全部一个值
    print(data_train['n11'].value_counts())#离散型变量，相差悬殊，用不用再分析
    print(data_train['n12'].value_counts())#离散型变量，相差悬殊，用不用再分析

#数值连续型变量分析
def serialValue():
    f = pd.melt(data_train, value_vars=numerical_serial_fea)#value_vars: needs to be transfered
    print(f)
    g = sns.FacetGrid(f, col="variable",  col_wrap=2, sharex=False, sharey=False)
    g.map(sns.distplot, "value")
    plt.show()

#Ploting Transaction Amount Values Distribution
def transactionAmount():
    plt.figure(figsize=(16,12))
    plt.suptitle('Transaction Values Distribution', fontsize=22)
    plt.subplot(221)
    sub_plot_1 = sns.distplot(data_train['loanAmnt'])
    sub_plot_1.set_title("loanAmnt Distribuition", fontsize=18)
    sub_plot_1.set_xlabel("")
    sub_plot_1.set_ylabel("Probability", fontsize=15)

    plt.subplot(222)
    sub_plot_2 = sns.distplot(np.log(data_train['loanAmnt']))
    sub_plot_2.set_title("loanAmnt (Log) Distribuition", fontsize=18)
    sub_plot_2.set_xlabel("")
    sub_plot_2.set_ylabel("Probability", fontsize=15)
    plt.show()

def singleValueVisualize():
    plt.figure(figsize=(8, 8))
    sns.barplot(data_train["employmentLength"].value_counts(dropna=False)[:20],
            data_train["employmentLength"].value_counts(dropna=False).keys()[:20])
    plt.show()
    
def noserialDiff():
    train_loan_fr = data_train.loc[data_train['isDefault'] == 1]
    train_loan_nofr = data_train.loc[data_train['isDefault'] == 0]
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 8))
    train_loan_fr.groupby('grade')['grade'].count()\
        .plot(kind='barh', ax=ax1, title='Count of grade fraud')
    train_loan_nofr.groupby('grade')['grade'].count()\
        .plot(kind='barh', ax=ax2, title='Count of grade non-fraud')
    train_loan_fr.groupby('employmentLength')['employmentLength'].count()\
        .plot(kind='barh', ax=ax3, title='Count of employmentLength fraud')
    train_loan_nofr.groupby('employmentLength')['employmentLength'].count()\
        .plot(kind='barh', ax=ax4, title='Count of employmentLength non-fraud')
    plt.show()

def serialDiff():
    fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(15, 6))
    data_train.loc[data_train['isDefault'] == 1] \
        ['loanAmnt'].apply(np.log) \
        .plot(kind='hist',
            bins=100,
            title='Log Loan Amt - Fraud',
            color='r',
            xlim=(-3, 10),
            ax= ax1)
    data_train.loc[data_train['isDefault'] == 0] \
        ['loanAmnt'].apply(np.log) \
        .plot(kind='hist',
            bins=100,
            title='Log Loan Amt - Not Fraud',
            color='b',
            xlim=(-3, 10),
            ax=ax2)
    plt.show()

def serialDiff2():
    total = len(data_train)
    total_amt = data_train.groupby(['isDefault'])['loanAmnt'].sum().sum()
    plt.figure(figsize=(12,5))
    plt.subplot(121)##1代表行，2代表列，所以一共有2个图，1代表此时绘制第一个图。
    plot_tr = sns.countplot(x='isDefault',data=data_train)#data_train‘isDefault’这个特征每种类别的数量**
    plot_tr.set_title("Fraud Loan Distribution \n 0: good user | 1: bad user", fontsize=14)
    plot_tr.set_xlabel("Is fraud by count", fontsize=16)
    plot_tr.set_ylabel('Count', fontsize=16)
    for p in plot_tr.patches:
        height = p.get_height()
        plot_tr.text(p.get_x()+p.get_width()/2.,
                height + 3,
                '{:1.2f}%'.format(height/total*100),
                ha="center", fontsize=15) 
        
    percent_amt = (data_train.groupby(['isDefault'])['loanAmnt'].sum())
    percent_amt = percent_amt.reset_index()
    plt.subplot(122)
    plot_tr_2 = sns.barplot(x='isDefault', y='loanAmnt',  dodge=True, data=percent_amt)
    plot_tr_2.set_title("Total Amount in loanAmnt  \n 0: good user | 1: bad user", fontsize=14)
    plot_tr_2.set_xlabel("Is fraud by percent", fontsize=16)
    plot_tr_2.set_ylabel('Total Loan Amount Scalar', fontsize=16)
    for p in plot_tr_2.patches:
        height = p.get_height()
        plot_tr_2.text(p.get_x()+p.get_width()/2.,
                height + 3,
                '{:1.2f}%'.format(height/total_amt * 100),
                ha="center", fontsize=15)  
    plt.show()

#时间格式数据处理
def dateValue():
    #转化成时间格式  issueDateDT特征表示数据日期离数据集中日期最早的日期（2007-06-01）的天数
    data_train['issueDate'] = pd.to_datetime(data_train['issueDate'],format='%Y-%m-%d')
    startdate = datetime.datetime.strptime('2007-06-01', '%Y-%m-%d')
    data_train['issueDateDT'] = data_train['issueDate'].apply(lambda x: x-startdate).dt.days
    plt.hist(data_train['issueDateDT'], label='train')
    plt.hist(data_test_a['issueDateDT'], label='test')
    plt.legend()
    plt.title('Distribution of issueDateDT dates')
    #train 和 test issueDateDT 日期有重叠 所以使用基于时间的分割进行验证是不明智的
    plt.show()

#
def index():
    pivot = pd.pivot_table(data_train, index = ['grade'], columns=['issueDateDT'], values = ['loanAmnt'], aggfunc = np.sum)
    print(pivot)

def genDataReport():
    pfr = pandas_profiling.ProfileReport(data_train)
    pfr.to_file("./example.html")

genDataReport()