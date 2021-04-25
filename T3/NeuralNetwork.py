import numpy as np
 #learning rate
 #regularization
 #layer number
 #test data comparison
 #outputs de 3 o mas, nuestra funcion de activacion es para si o no
class NeuralNetwork:
    def __init__(self, x, y, numLayers=4):
        self.numLayers        = numLayers
        self.input            = x
        self.neuronsPerLayer  = x.shape[0]
        self.weightsInitial   = np.random.rand(x.shape[1],self.neuronsPerLayer) 
        self.weights = [np.random.rand(self.neuronsPerLayer,self.neuronsPerLayer) for i in range(1,numLayers)]
        self.weightsFinal     = np.random.rand(self.neuronsPerLayer,1)                 
        self.y                = y
        self.output           = np.zeros(y.shape)
    # Activation function
    def sigmoid(self,z):
        return 1/(1+np.exp(-z))

    # Derivative of sigmoid
    def sigmoid_derivative(self,p):
        return p * (1 - p)

    def feedforward(self):#for 
        self.layer1 = self.sigmoid(np.dot(self.input, self.weightsInitial))
        self.layer2 = self.sigmoid(np.dot(self.layer1, self.weightsFinal))
        return self.layer2

    def backprop(self):
        # application of the chain rule to find derivative of the loss function with respect to weightsFinal and weightsInitial
        d_weightsFinal = np.dot(self.layer1.T, (2*(self.y - self.output) * self.sigmoid_derivative(self.output)))
        d_weightsInitial = np.dot(self.input.T,  (np.dot(2*(self.y - self.output) * self.sigmoid_derivative(self.output), self.weightsFinal.T) * self.sigmoid_derivative(self.layer1)))

        # update the weights with the derivative (slope) of the loss function
        self.weightsInitial += d_weightsInitial
        self.weightsFinal += d_weightsFinal

    def train(self, X, y):
        self.output = self.feedforward()
        self.backprop()

    def loss(self, yReal):
        return np.mean(np.square(yReal - self.feedforward()))

    def accuracy(self, yReal):
        predictions = np.array([1 if i > 0.5 else 0 for i in self.feedforward()])
        results = np.array([1 if predictions[i] == yReal[i] else 0 for i in range(len(yReal))])

        return np.mean(results)