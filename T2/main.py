import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random


from sklearn.datasets import load_breast_cancer
# Grafica el la funcion de costo
def cost_plot(epochs, cost_hist):
    plt.subplots(figsize=(12,8))
    plt.ylabel('Cost')
    plt.xlabel('Iterations')
    plt.plot(range(epochs),cost_hist,'b.')
    plt.show()

# esto es para hacer el scatter plot con la linea resultante
def graficar_datos(x, y, linea):
    plt.figure(figsize = (10, 6))
    plt.scatter(x, y, color = 'green')
    plt.plot(x , linea , color = 'k' , lw = 3)
    plt.title("Salinity and temperature features in thebottledataset by CalCOFI")
    plt.ylabel('temperature', size = 20)
    plt.xlabel('salinity', size = 20)
    plt.show()

def sigmoid_function(x):
    #funcion sigmoide para calcular la hipotesis
    return 1/(1+np.exp(-x))

# aplica descenso del gradiante
def gradient_descent(x, y, theta, epochs=2000, alpha=0.0001):
    n = float(len(y))
    cost_hist = [] # guarda datos para graficar la funcion de costo
    for i in range(epochs):

        #prediccion de linea
        #y_pred = theta0 + (theta1 * x)
        y_pred = sigmoid_function(x)
        
        #derivadas
        derivada_theta0 = (1.0/n) * sum(y_pred - y)
        derivada_theta1 = (1.0/n) * sum((y_pred - y) * x)
        
        #actualizar thetas
        theta[0] = theta[0] - (alpha * derivada_theta0)
        theta[1] = theta[1] - (alpha * derivada_theta1)

        if(i%1000==0):
            print("\n{2}\n T0: {0}\n T1: {1}\n".format(theta[0], theta[1], i))
        cost_hist.append((1/(2*n)) * np.sum(np.square(y_pred - y)))
        
    print("\n{2}\n T0: {0}\n T1: {1}\n".format(theta[0], theta[1], i))
    return theta[0], theta[1], cost_hist



def cost_function():
    pass


def scatter_plot(X,Y):
    area = np.pi*3
    red = "#ff0000"
    green = "#00ff00"
    colors = [red if i == 0 else green for i in Y]
    
    plt.xlabel('test1')
    plt.ylabel('test2')
    
    plt.scatter(X[0], X[1], s=area, c=colors)
    plt.show()

def Logistic_Regression(X,Y,alpha):
    pass

def loss(h, y):
    return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()

def gradient_descent(X, h, y):
    return np.dot(X.T, (h - y)) / y.shape[0]

def main():
    df = pd.read_csv("dataset-2.csv")
    #shuffle DataFrame rows
    shuffle_df = df.sample(frac=1)
    train_size = int(0.7 * len(df))
    train_set = shuffle_df[:train_size]
    test_set = shuffle_df[train_size:]

    XTr, XTe = np.array(train_set[train_set.columns[0:2]]), np.array(test_set[test_set.columns[0:2]])
    YTr, YTe = np.array(train_set['accepted']), np.array(test_set['accepted'])
    XTe = np.transpose(XTe)
    
    print(XTe.shape)
    # print(YTe)
    scatter_plot(XTe,YTe)

    theta = np.zeros(XTr.shape[0])
    # print(theta)
  

def main2():
    #Loading the data
    data = load_breast_cancer()

    #Preparing the data
    x = data.data
    y = data.target
    print(y)


if __name__ == "__main__":
    main()