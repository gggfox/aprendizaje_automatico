import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statistics import mean
from sklearn.linear_model import LinearRegression

def graficar_datos(x, y, linea):
    #linea = (-4.8 * x) + 169.11
    plt.figure(figsize = (10, 6))
    plt.scatter(x, y, color = 'green')
    plt.plot(x , linea , color = 'k' , lw = 3)
    plt.ylabel('temperature', size = 20)
    plt.xlabel('salinity', size = 20)
    #plt.axis([0,40,0,40])
    plt.show()
   
def cost_plot(epochs, cost_hist):
    plt.subplots(figsize=(12,8))
    plt.ylabel('Cost')
    plt.xlabel('Iterations')
    plt.plot(range(epochs),cost_hist,'b.')
    plt.show()
    
def gradient_descent(x, y, epochs=2000, alpha=0.0001, theta0=0, theta1=0):
    n = float(len(y))
    cost_hist = []
    for i in range(epochs):
        #prediccion de linea
        y_pred = theta0 + (theta1 * x)
        
        #derivadas
        derivada_theta0 = (1.0/n) * sum(y_pred - y)
        derivada_theta1 = (1.0/n) * sum((y_pred - y) * x)
        
        #actualizar thetas
        theta0 = theta0 - (alpha * derivada_theta0)
        theta1 = theta1 - (alpha * derivada_theta1)

        if(i%1000==0):
            print("\n{2}\n T0: {0}\n T1: {1}\n".format(theta0, theta1, i))
        cost_hist.append((1/(2*n)) * np.sum(np.square(y_pred - y)))
        
    print("\n{2}\n T0: {0}\n T1: {1}\n".format(theta0, theta1, i))
    cost_plot(epochs, cost_hist)
    return theta0, theta1
        
def main():
    df = pd.read_csv('temp.csv')
    Y  = np.array(df['temperature'])
    X  = np.array(df['salinity'])
    theta0, theta1 = gradient_descent(X, Y)
    y_pred = theta0 + (theta1 * X)
    graficar_datos(X, Y, y_pred)
     
        
main()
