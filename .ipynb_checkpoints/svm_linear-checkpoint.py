import matplotlib.pyplot as plt
import sklearn.datasets as skdata
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics

numeros = skdata.load_digits()
target = numeros['target']
imagenes = numeros['images']
n_imagenes = len(target)


data = imagenes.reshape((n_imagenes, -1)) # para volver a tener los datos como imagen basta hacer data.reshape((n_imagenes, 8, 8))
print(np.shape(data))

scaler = StandardScaler()
x_2, x_validation, y_2, y_validation = train_test_split(data, target, train_size=0.8)
x_train, x_test, y_train, y_test = train_test_split(x_2, y_2, train_size=0.5)

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
x_validation = scaler.transform(x_validation)

cov = np.cov(x_train.T)
valores, vectores = np.linalg.eig(cov)
valores = np.real(valores)
vectores = np.real(vectores)
ii = np.argsort(-valores)
valores = valores[ii]
vectores = vectores[:,ii]

proyeccion_train = np.matmul(x_train,vectores)[:,:30]
proyeccion_test = np.matmul(x_test,vectores)[:,:30]
proyeccion_validation = np.matmul(x_validation,vectores)[:,:30]

hiperparametros = np.logspace(-3,2,20)
scores = []

for C in hiperparametros:
    clasificador = SVC(C=C)
    clasificador.fit(proyeccion_train,y_train)
    predicciones = clasificador.predict(proyeccion_test)
    scores.append(metrics.f1_score(y_test,predicciones,average='macro'))
    

plt.figure()
plt.plot(hiperparametros,scores)
plt.xscale('log')
plt.scatter(hiperparametros[np.argmax(scores)],np.amax(scores))

mejor_C = hiperparametros[np.argmax(scores)]
print(mejor_C)


clasificador = SVC(C=mejor_C)
clasificador.fit(proyeccion_train,y_train)
predicciones = clasificador.predict(proyeccion_validation)

matriz = metrics.confusion_matrix(y_validation,predicciones)


plt.figure(figsize=(8,8))
plt.imshow(matriz)

for i in range(0,10):
    for j in range(0,10):
        plt.text(i-0.5,j,' {:.2f}'.format(float(matriz[i,j])/np.sum(y_validation==i)),fontsize=10)
        
plt.title('C = {:.2f}'.format(mejor_C))
plt.axis('off')
plt.title('C = {:.2f}'.format(mejor_C))
plt.axis('off')
plt.savefig('matriz_de_confusion.png')