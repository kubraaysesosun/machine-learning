# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 02:42:08 2018

@author: AYSE
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#verilerin yukleme
veriler= pd.read_csv("veriler.csv")

print(veriler)

#veri on isleme
boy= veriler[["boy"]]
print(boy)

x=10
class insan:
    boy=130
    def kosmak(self,b):
        return b+10
    


#eksik veriler

from sklearn.preprocessing import Imputer
imputer=Imputer(missing_values='NaN', strategy='mean', axis=0)
Yas=veriler.iloc[:,1:4].values

imputer=imputer.fit(Yas[:,1:4])
Yas[:,1:4]=imputer.transform(Yas[:,1:4])



#kategorik verileri sayısala cevirme

ulke= veriler.iloc[:,0:1].values
print(ulke)

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le = LabelEncoder()
ulke[:,0] =le.fit_transform(ulke[:,0])
print(ulke)
#from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(categorical_features='all')
ulke = ohe.fit_transform(ulke).toarray()
print(ulke)

c= veriler.iloc[:,-1:].values
print(c)

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le = LabelEncoder()
c[:,0] =le.fit_transform(c[:,0])
print(c)
#from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(categorical_features='all')
c = ohe.fit_transform(c).toarray()
print(c)

#verilerin birleştirilmesi ve dataframe oluşturulması
print(list(range(22)))
sonuc = pd.DataFrame(data=ulke, index=range(22), columns=['fr','tr','us'])  
print(sonuc)
sonuc2 = pd.DataFrame(data=Yas, index=range(22), columns=['boy','kilo','yas'])  
print(sonuc2)

cinsiyet=veriler.iloc[:,-1].values
print(cinsiyet)

sonuc3=pd.DataFrame(data=c[:,:1],index=range(22),columns=['cinsiyet'])
print(sonuc3)

s=pd.concat([sonuc,sonuc2],axis=1)
print(s)
s2=pd.concat([s,sonuc3],axis=1)
print(s2)

#veri kumesinin egitim ve test olarak ayrilmasi
from sklearn.cross_validation import train_test_split #farklı bolme islemleride var
x_train, x_test,y_train,y_test=train_test_split(s,sonuc3,test_size=0.33,random_state=0)


from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)#x_trainden y_traini öğren

y_pred=regressor.predict(x_test)
  
boy=s2.iloc[:,3:4].values
print(boy)
#veri kumesine inşaa etmek için
sol=s2.iloc[:,:3]
sag=s2.iloc[:,4:]
veri=pd.concat([sol,sag],axis=1 )

x_train, x_test,y_train,y_test=train_test_split(veri,boy,test_size=0.33,random_state=0)


r2=LinearRegression()
r2.fit(x_train,y_train)#x_trainden y_traini öğren

y_pred=r2.predict(x_test)


#backward elimination için şablon
import statsmodels.formula.api as sm
X =np.append(arr=np.ones((22,1)).astype(int), values=veri, axis=1)
X_l=veri.iloc[:,[0,1,2,3,4,5]].values
r_ols=sm.OLS(endog=boy,exog=X_l)
r=r_ols.fit()
print(r.summary())
#summary de çıkan x1,2,3,4,5,6 değerlerine bakarak en yüksek p değerini eliyeceğiz. (x5)

X_l=veri.iloc[:,[0,1,2,3,5]].values
r_ols=sm.OLS(endog=boy,exog=X_l)
r=r_ols.fit()
print(r.summary())

X_l=veri.iloc[:,[0,1,2,3]].values
r_ols=sm.OLS(endog=boy,exog=X_l)
r=r_ols.fit()
print(r.summary())





































