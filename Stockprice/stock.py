# -*- coding:gbk-*-
import numpy as np
import tushare as ts
import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt

#定义list
stocklist = pd.read_csv('1.txt')  #stocklist
datelist = ['2017-03-15','2017-03-16']  #datelist
analysispar = ['close','volume','ma5','ma10','ma20','v_ma5','v_ma10','v_ma20','turnover','nextclose']  #参数list
#每个参数分配一个LIST，并拼接为一个LIST
stockname = []
nextclose= []
close = []
volume = []
ma5 = []
ma10 = []
ma20 = []
v_ma5 = []
v_ma10 = []
v_ma20 = []
turnover = []
parlist = [close,volume,ma5,ma10,ma20,v_ma5,v_ma10,v_ma20,turnover]


#循环股票列表
for f in range(len(stocklist)):
    print str(f+1)+"/"+str(len(stocklist))
    a = ts.get_hist_data(str(stocklist.ix[f][0]),start=datelist[0],end=datelist[1])  #获取数据
    if (len(a) == 2):
        stockname.append(str(stocklist.ix[f][0]))  #单独处理股票名LIST
        for i in range(len(analysispar)-1):
            parlist[i].append(a.ix[datelist[0],analysispar[i]])  #每个数据存入相应LIST
        nextclose.append(a.ix[datelist[1],'close'])  #单独处理第二天收盘LIST

#画图
b = False
if (b == True): 
    plt.figure()
    for w in range(len(parlist)):
        plt.subplot(3,3,w+1)
        plt.plot(parlist[w], nextclose)
    plt.show()



#形成数据dict
data = {'close':close,
        'volume':volume,
        'ma5':ma5,
        'ma10':ma10,
        'ma20':ma20,
        'v_ma5':v_ma5,
        'v_ma10':v_ma10,
        'v_ma20':v_ma20,
        'turnover':turnover,
        'nextclose':nextclose}
#创建dataframe
stockframe = DataFrame(data,columns=analysispar,index=stockname)


#print stockframe
X = stockframe[['close','volume','ma5','ma10','ma20','v_ma5','v_ma10','v_ma20','turnover']]
y = stockframe['nextclose']

from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
linreg.fit(X, y)
print linreg.intercept_
print linreg.coef_
print zip(['close','volume','ma5','ma10','ma20','v_ma5','v_ma10','v_ma20','turnover'], linreg.coef_)



a = False
if (a == True):
    ##测试集和训练集的构建
    from sklearn.cross_validation import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    linreg.fit(X_train, y_train)
    #结果
    print linreg.intercept_
    print linreg.coef_
    print zip(['close','volume','ma5','ma10','ma20','v_ma5','v_ma10','v_ma20','turnover'], linreg.coef_)

    #预测
    y_pred = linreg.predict(X_test)

    #误差评估
    from sklearn import metrics

    # calculate MAE using scikit-learn
    print "MAE:",metrics.mean_absolute_error(y_test,y_pred)

    # calculate MSE using scikit-learn
    print "MSE:",metrics.mean_squared_error(y_test,y_pred)

    # calculate RMSE using scikit-learn
    print "RMSE:",np.sqrt(metrics.mean_squared_error(y_test,y_pred))



#开始测试
def cal(list):  #定义计算公式
    res = 0
    for i in range(len(linreg.coef_)):
        res = res + list[i]*linreg.coef_[i]
    return res + linreg.intercept_

#读取测试数据
stocklisttest = pd.read_csv('2.txt')  #stocklist
rightcount = 0
totalcount = 0
for f in range(len(stocklisttest)):
    testlist = []
    print str(f+1)+"/"+str(len(stocklisttest))
    a = ts.get_hist_data(str(stocklisttest.ix[f][0]),start=datelist[0],end=datelist[1])  #获取数据
#    print a
    if (len(a) == 2):
        totalcount = totalcount+1
        for i in range(len(analysispar)-1):
            testlist.append(a.ix[datelist[0],analysispar[i]])
        testlist.append(a.ix[datelist[1],'close'])
        calresult = cal(testlist)
        if (calresult > testlist[0]) and (testlist[-1] > testlist[0]) or (calresult < testlist[0]) and (testlist[-1] < testlist[0]):
            rightcount = rightcount+1

print str(rightcount)+"/"+str(totalcount)
