print("Hello ml ")

import pandas as pd #Verileri düzgün bir şekilde tutmak için bu kütüphaneyi kullanırız
import numpy as np  # Büyük sayılar venumeric işlemler için kullanılır 
import matplotlib.pyplot as plt # Çizimler için kullanılır.

#kodlar 
#veri yükleme
veriler = pd.read_csv('veriler.csv')

print(veriler)

x= veriler.iloc[:,1:4].values
y= veriler.iloc[:,4:].values




from sklearn.model_selection import train_test_split

x_train, x_test ,y_train,y_test = train_test_split(x,y,test_size=0.33,random_state=0)
#x bizim bağımsız değişkenimiz. y hedef bağımlı değişkendir.


from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test) # x test için yeniden öğrenme,öğrenmiş olduğun x trainden tranform  et

from sklearn.linear_model import LogisticRegression
#obje oluşturma:
logr = LogisticRegression(random_state=0)
logr.fit(X_train, y_train) #eğitiyor

y_pred = logr.predict(X_test)
print(y_pred)
print(y_test)


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)


















""


