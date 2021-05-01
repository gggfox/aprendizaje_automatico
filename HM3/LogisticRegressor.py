import numpy as np

class LogisticRegressor():
    def __init__(self, alpha=0.1, epochs=1, regularize=False, reg_factor=0.1):
        self.alpha = alpha
        self.epochs = epochs
        self.regularize = regularize
        self.reg_factor = reg_factor
        self.costs = []
        self.theta = []
   
    def _cost_function(self, hyp, y):
        # returns a scalar
        m = len(y)
        class_1_part = y*np.log(hyp)
        class_0_part = (1-y)*np.log(1-hyp)
        cost = (-1.0/m) * np.sum( class_1_part + class_0_part)
        if self.regularize:
            all_but_theta_0 = self.theta[1:,:]
            cost += self.reg_factor * np.sum((all_but_theta_0**2)) /(2*m)
        return cost
    
    def _hypothesis(self, theta, X):
        # * is element wise multiplication
        # numpy.dot(), or @ operator will work
        return 1.0 / (1 + np.exp(-theta.T @ X))
    
    def fit(self, X, y):
        n,m = X.shape[0], X.shape[1]
        self.theta = np.array([np.zeros(n)]).T # This is to get a nx1 array
        # np.random.seed(0)
        # self.theta =  np.random.uniform(-10,10,(n,1))

        # rand_start = np.random.rand(n)
        # self.theta = rand_start.reshape(rand_start.shape[0],-1) # This is to get a nx1 array
        for i in range(self.epochs):
            # X is (n x m), y is (1 x m), theta is (nx1)
            hyp = self._hypothesis(self.theta, X)   # hyp is (1xm) vector
            # cost = self._cost_function(hyp, y)      # cost is a scalar
            # self.costs.append(cost)

            if self.regularize:
                gradient_theta_0 =  (X[0,:] @ (hyp - y).T)            # (1xm) x (mx1) = scalar
                gradient_all_but_theta_0 = ( (X[1:, :] @ (hyp - y).T) + (self.reg_factor/m) * self.theta[1:, :] )   # ((n-1)xm) x (mx1) = (n-1)x1
                self.theta[0] -= self.alpha * (gradient_theta_0/m) # the scalar we just got
                self.theta[1:, :] -= self.alpha * (gradient_all_but_theta_0/m) # the (n-1)x1 we just got
            else:
                gradient = (X @ (hyp - y).T) / m   # (nxm) x (mx1) = (nx1) with the gradient per theta (dimension)
                self.theta -= self.alpha * gradient        # new theta values
            # print("Cost iter {} : {}".format(i, cost)); # it is helpful to print your cost function to see what's going on with GD convergence

        print("Final theta is {}".format(self.theta.T))
    
    def predict(self, X):
        return np.where(self._hypothesis(self.theta, X) > 0.5, 1, 0)
        # return self._hypothesis(self.theta, X) >= 0.5