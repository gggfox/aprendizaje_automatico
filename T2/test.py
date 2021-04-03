import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
import pandas as pd
import random as rand
from logistic_regression import LogisticRegression

#funcion para calcular accuracy de valores reales contra predicciones
def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

#fucnion para crear scatter plot de los datos
def scatter_plot(X,Y):
    area = np.pi*3
    red = "#ff0000"
    green = "#00ff00"
    colors = [red if i == 0 else green for i in Y]
    
    plt.scatter(X[1], X[2], s=area, c=colors)
    plt.show()

def main():
    #############################################
    #--VALORES INICIALES PARA WEIGHTS & BIASES--#
    #############################################
    #dataset1 1.0                               #
    h0=1                                        #
    weights = [h0, 0.04511972, 0.03886596]      #
    bias = -5.010006883123912                   #
    #dataset2 0.778                             #
    # weights = [h0,-0.23544893 ,-0.04018184]   #
    # bias = 0.01917428134188063                #
    # #dataset2 0.806778                        #
    # weights = [h0,-0.19985231, -0.06794084]   #
    # bias = 0.052870257825161665               #
    # #dataset2 0.833                           #
    weights = [h0,-0.23877248, -0.04799755]     #
    bias = 0.04149726514161977                  #
    #############################################


    #hyperparametros de modelo
    iters = 3000 #iteraciones de aprendizaje
    lr = 0.00001 #learning rate
    reg = False #booleano para decidir si se usa regularizacion o no

    #lectura de archivos csv
    df = pd.read_csv("dataset-2.csv")
    df2 = pd.read_csv("dataset-2-modified.csv")
    df2 = df2.join(df['accepted'])
    train_size = int(0.7 * len(df))

    # valores para ciclo del programa
    best_regressor = 0

    acc = 0.0
    prev_acc = 0.0
    cont = 0
    acc_goal = 0.97
    max_cont = 200
    #ciclo de iteraciones de modelo
    while (acc < acc_goal and cont < max_cont):
        
        #elegir valores aleatorios para cada generacion de modelo
        shuffle_df = df.sample(frac=1)
        train_set = shuffle_df[:train_size]
        test_set = shuffle_df[train_size:]

        #separar sets
        X_train, X_test = np.array(train_set[train_set.columns[0:27]]), np.array(test_set[test_set.columns[0:27]])
        #X_train, X_test = np.insert(X_train,0,1,axis=1), np.insert(X_test,0,1,axis=1)    
        y_train, y_test = np.array(train_set['accepted']), np.array(test_set['accepted'])
            
        #crear modelo 
        regressor = LogisticRegression(weights, bias, lr, iters)
        #decenso de gradiante
        regressor.fit(X_train, y_train, reg)
        #genracion de predicicones
        predictions = regressor.predict(X_test)

        #medir precicon del modelo
        acc = accuracy(y_test, predictions)
        cont += 1

        print("LR classification accuracy:{0:.3f}".format(acc))
        print("\n ({0:.3f} > {1:.3f} ); lr: {2:.6f}; cont: {3}\n".format(acc,prev_acc,lr,cont))
        if(acc >= prev_acc):
            cont = 0
            prev_acc = acc
            #guardar datos del mejor modelo
            weights = regressor.weights
            bias = regressor.bias
            best_regressor = regressor

    #impirmir datos del mejor modelo
    print(weights)
    print(bias)

    scatter_plot(X_test.T,best_regressor.predict(X_test))    

if __name__ == '__main__':
    main()