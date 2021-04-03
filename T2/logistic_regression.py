import numpy as np

class LogisticRegression:
    
    #hiperparametros
    def __init__(self, weights, bias, learning_rate, n_iters):

        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = weights #theta
        self.bias = bias #theta0
        self.reg = 0

    # funcion para minimizar el error entre valores reales de la variable target y los valores de prediccion de la misma
    def cost(self,N,X,diff):
        #derrivadas
        dw = (1 / N) * np.dot(X.T, diff) + self.reg
        db = (1 / N) * np.sum(diff)
        #actualizacion de thetas (weights y bias)
        self.weights -= self.lr * dw
        self.bias -= self.lr * db

    #funcion para generacion de modelo
    def fit(self, X, y, isReg):
        N, _ = X.shape #cantidad de datos
        #fucnion para verificar si se neceista user regularizacion
        self.reg = ((np.mean(X)/N) * np.sum(self.weights)) if isReg else 0

        # gradient descent
        for _ in range(self.n_iters):
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = self._sigmoid(linear_model)
            diff = y_pred - y

            self.cost(N,X,diff)

        print("weights: {0} \nbias: {1}".format(self.weights,self.bias))

    #mehodo para prediccion de datos testing
    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self._sigmoid(linear_model)
        y_predicted_cls = [1 if i >= 0.5 else 0 for i in y_predicted]
        return np.array(y_predicted_cls)
    
    #funcion sigmoida para regession logistica
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))