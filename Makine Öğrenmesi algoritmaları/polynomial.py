#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score

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

print("Linear R2 Değeri")
print(r2_score(Y,lin_reg2.predict(poly_reg.fit_transform(X))))


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

print("Polynomial R2 değeri")
print(r2_score(Y,lin_reg2.predict(poly_reg.fit_transform(X))))


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

plt.show()
print(svr_reg.predict([[6.6]]))

print("Decision Tree R2 değeri")
print(svr_reg.predict(x_olcekli))

#Decision Tree Regression
from sklearn.tree import DecisionTreeRegressor
r_dt = DecisionTreeRegressor(random_state= 0)
r_dt.fit(X,Y) #X'den Y yi öğren
Z= X +0.5
K= X - 0.4

plt.scatter(X,Y,color='red')
plt.plot(X, r_dt.predict(X),color='blue')

plt.plot(x,r_dt.predict(Z),color= 'green')
plt.plot(x,r_dt.predict(K),color='yellow')
plt.show()
print(r_dt.predict([[6.6]]))

#Random Forest Regression
from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor(n_estimators=10, random_state=0)
rf_reg.fit(X,Y.ravel())

print(rf_reg.predict([[6.5]]))

plt.scatter(X,Y,color='red')
plt.plot(X,rf_reg.predict(X),color='green')

plt.plot(X,rf_reg.predict(Z),color='green')
plt.plot(x,r_dt.predict(K),color='yellow')
plt.show()


#R2
print("Random forest R2 Değeri")
print(r2_score(Y,rf_reg.predict(X)))
print(r2_score(Y,rf_reg.predict(K)))
print(r2_score(Y,rf_reg.predict(Z)))


print("Linear R2 Değeri")
print(r2_score(Y,lin_reg2.predict(poly_reg.fit_transform(X))))

print("Polynomial R2 değeri")
print(r2_score(Y,lin_reg2.predict(poly_reg.fit_transform(X))))


print("SVR R2 Değeri")
print(r2_score(y_olcekli,svr_reg.predict(x_olcekli)))

print("Decision Tree R2 değeri")
print(svr_reg.predict(x_olcekli))

print("Random forest R2 Değeri")
print(r2_score(Y,rf_reg.predict(X)))












