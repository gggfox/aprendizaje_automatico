import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from NeuralNetwork import NeuralNetwork

#file = input("Give me the name of the file")
#df = pd.read_csv("{0}.csv".format(file))
df = pd.read_csv("xor.csv")
df = df.sample(frac=1)#randomize dataset
(rows, columns) = df.shape
trainSize = int(rows * 0.75)

xTrain = np.array(df.iloc[:trainSize,:columns-1])
xTest  = np.array(df.iloc[trainSize:,:columns-1])
yTrain = np.array(df.iloc[:trainSize,columns-1:])
yTest  = np.array(df.iloc[trainSize:,columns-1:])

x=np.array(([0,0,1],[0,1,1],[1,0,1],[1,1,1]), dtype=float)
y=np.array(([0],[1],[1],[0]), dtype=float)

print(xTrain.shape)
print(yTrain.shape)

if(1==0):
    x=xTrain
    y=yTrain

NN = NeuralNetwork(x,y)
def TrainModel(NN,epochs = 3000):
    for i in range(epochs): # trains the NN 1,000 times
        if i % 100 ==0: 
           
            print ("for iteration # " + str(i) + "\n")
            print ("Input : \n" + str(x))
            print ("Actual Output: \n" + str(y))
            print ("Predicted Output: \n" + str(NN.feedforward()))
            print ("Loss: \n" + str(NN.loss(y))) # mean sum squared loss
            print ("Accuracy: \n" + str(NN.accuracy(y))) # mean sum squared los
            print ("\n")
    
        NN.train(x, y)
    print ("Accuracy: \n" + str(NN.accuracy(y))) # mean sum squared los)
    print("final weights: \n" + str(NN.weightsFinal))


TrainModel(NN)
print(x.shape)
print(y.shape)