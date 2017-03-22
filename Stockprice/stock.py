# -*- coding:gbk-*-
import numpy as np
import tushare as ts
import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt

#����list
stocklist = pd.read_csv('1.txt')  #stocklist
datelist = ['2017-03-15','2017-03-16']  #datelist
analysispar = ['close','volume','ma5','ma10','ma20','v_ma5','v_ma10','v_ma20','turnover','nextclose']  #����list
#ÿ����������һ��LIST����ƴ��Ϊһ��LIST
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


#ѭ����Ʊ�б�
for f in range(len(stocklist)):
    print str(f+1)+"/"+str(len(stocklist))
    a = ts.get_hist_data(str(stocklist.ix[f][0]),start=datelist[0],end=datelist[1])  #��ȡ����
    if (len(a) == 2):
        stockname.append(str(stocklist.ix[f][0]))  #���������Ʊ��LIST
        for i in range(len(analysispar)-1):
            parlist[i].append(a.ix[datelist[0],analysispar[i]])  #ÿ�����ݴ�����ӦLIST
        nextclose.append(a.ix[datelist[1],'close'])  #��������ڶ�������LIST

#��ͼ
b = False
if (b == True): 
    plt.figure()
    for w in range(len(parlist)):
        plt.subplot(3,3,w+1)
        plt.plot(parlist[w], nextclose)
    plt.show()



#�γ�����dict
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
#����dataframe
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
    ##���Լ���ѵ�����Ĺ���
    from sklearn.cross_validation import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    linreg.fit(X_train, y_train)
    #���
    print linreg.intercept_
    print linreg.coef_
    print zip(['close','volume','ma5','ma10','ma20','v_ma5','v_ma10','v_ma20','turnover'], linreg.coef_)

    #Ԥ��
    y_pred = linreg.predict(X_test)

    #�������
    from sklearn import metrics

    # calculate MAE using scikit-learn
    print "MAE:",metrics.mean_absolute_error(y_test,y_pred)

    # calculate MSE using scikit-learn
    print "MSE:",metrics.mean_squared_error(y_test,y_pred)

    # calculate RMSE using scikit-learn
    print "RMSE:",np.sqrt(metrics.mean_squared_error(y_test,y_pred))



#��ʼ����
def cal(list):  #������㹫ʽ
    res = 0
    for i in range(len(linreg.coef_)):
        res = res + list[i]*linreg.coef_[i]
    return res + linreg.intercept_

#��ȡ��������
stocklisttest = pd.read_csv('2.txt')  #stocklist
rightcount = 0
totalcount = 0
for f in range(len(stocklisttest)):
    testlist = []
    print str(f+1)+"/"+str(len(stocklisttest))
    a = ts.get_hist_data(str(stocklisttest.ix[f][0]),start=datelist[0],end=datelist[1])  #��ȡ����
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
