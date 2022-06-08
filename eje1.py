import random
import math
from sklearn import datasets #solo se usa la libreria para cargar iris

iris = datasets.load_iris()

def obtenerPesos(n):
    pesos = []
    for i in range(n):
        pesos.append(random.uniform(0, 1))
    return pesos

def productoPunto(x,y):
    sum = 0
    for (i, j) in zip(x,y):
        sum += i*j
    return sum

def sigmoide(x):
    return 1 / math.exp(-x)

def prediccion(entradas, pesos, yd, bias = 1):
    for  i, entrada in enumerate(entradas):
        y = productoPunto(entrada, pesos) + bias #capa1
        capa2 = sigmoide(y)
        print('y:',  y)
        print('sigmoide:',  capa2)
        if y>=0:
            y = 1
        else:
            y = -1
        error = yd[i] - y
        print('error: ', error)


entradas = iris.data

pesos = obtenerPesos(entradas[0].size)

punto = productoPunto(entradas, pesos)

prediccion(entradas, pesos, iris.target)



