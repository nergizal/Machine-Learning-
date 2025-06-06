#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# veri yukleme
veriler = pd.read_csv('maaslar.csv')

x = veriler.iloc[:,1:2]
y = veriler.iloc[:,2:]
X = x.values
Y = y.values


#linear regression
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,Y)

plt.scatter(X,Y,color='red')
plt.plot(x,lin_reg.predict(X), color = 'blue')
plt.show()


#polynomial regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
x_poly = poly_reg.fit_transform(X)
print(x_poly)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)
plt.scatter(X,Y,color = 'red')
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.show()

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
x_poly = poly_reg.fit_transform(X)
print(x_poly)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)
plt.scatter(X,Y,color = 'red')
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.show()

#tahminler

print(lin_reg.predict([[11]]))
print(lin_reg.predict([[6.6]]))

print(lin_reg2.predict(poly_reg.fit_transform([[6.6]])))
print(lin_reg2.predict(poly_reg.fit_transform([[11]])))




#verilerin ölçeklenmesi
from sklearn.preprocessing import StandardScaler

#fit_transform iki işi birden yapar: önce veriye göre öğrenir (fit), sonra dönüştürür (transform)
#X ve Y verileri ayrı ayrı ölçekleniyor.
#X genellikle giriş (bağımsız değişken),
#Y çıkış (bağımlı değişken) olur.
sc1= StandardScaler()
x_olcekli = sc1.fit_transform(X)
sc2 = StandardScaler()
y_olcekli = sc2.fit_transform(Y)

#2. SVR Modelinin Kurulması ve Eğitilmesi
from sklearn.svm import SVR
#SVR, yani Support Vector Regression, SVM’nin regresyon versiyonudur.
svr_reg = SVR(kernel = 'rbf')  # rbf = radial basis function, yani Gauss tipi çekirdek,kernel='rbf' sayesinde doğrusal olmayan ilişkileri de öğrenebilir.
svr_reg.fit(x_olcekli, y_olcekli) #fit(...) ile model, ölçeklenmiş veriler üzerinde eğitiliyor.

plt.scatter(x_olcekli, y_olcekli,color='red') #plt.scatter: Gerçek verileri noktalarla gösterir.
plt.plot(x_olcekli, svr_reg.predict(x_olcekli),color= 'blue') #plt.plot: Modelin tahmin ettiği eğriyi çizer.

print(svr_reg.predict(11.0))
print(svr_reg.predict(6.6))













