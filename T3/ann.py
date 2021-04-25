import numpy as np
import pandas as pd

def fit(X, y, alpha, reg_factor, epochs):
    pass

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def z(input, Theta):
    # we need to get an n x m array, where n is the neurons in this layer and m is the number of activations we used
    return Theta @ input

def activation(z):
    return sigmoid(z)

def initialize_weights(X, num_classes, hidden):
    weights = []
    out_neurons = num_classes # len(y)
    n = X.shape[0]

    h_layers = len(hidden)
    s_j = n
    # [2]
    for j in range(0,h_layers):
        s_jplus1 = hidden[j]
        cols = s_j + 1
        # print('Theta {} will be {}x{}'.format(j+1, s_jplus1, cols))
        # weights_layer = np.zeros((s_jplus1, cols))
        weights_layer = np.random.rand(s_jplus1, cols)
        weights.append(weights_layer)
        s_j = s_jplus1

    # weights.append(np.zeros((out_neurons, s_j + 1)))
    weights.append(np.random.rand(out_neurons, s_j + 1))
    return weights

def print_matrix(matrix):
    for e in matrix:
        print('shape: {} {}'.format(e.shape, e))

def initialize_activations(X, output_neurons, hidden):
    # initialize activations, they are
    # input layer: (n+1) x 1,
    # hidden: (each + 1) x 1
    # output: output_neurons x 1
    # So we can represent them with a list of length 2 + len(hidden)

    activations = []
    # input layer
    a_1 = np.array([X[:,0]]).T
    biases = np.ones(a_1.shape[1])
    a_1 = np.vstack((biases, a_1))
    activations.append(a_1)

    for i in range(len(hidden)):
        a_i = np.zeros((hidden[i] + 1 , 1))
        a_i[0,0] = 1.0
        activations.append(a_i)

    # output layer
    activations.append(np.zeros((output_neurons,1)))
    return activations

def forward(X, hidden, activations, weights):
    m = X.shape[1]
    for e in range(m):
        # a^0 = X
        # FeedForward
        # Hidden layers
        for i in range(0, len(hidden)):
            a_i = activations[i]
            # print('In Layer {}'.format(i+1))
            z_next = z(a_i, theta[i])
            # print('z_i are {}'.format(z_next.T))
            a_next = activation(z_next)
            activations[i+1][1:] = a_next # this line would fail for output layer
            # print('activations {} are {}'.format(i+1, a_next))

        # output layer (i+2 is the output layer)
        a_i = activations[i+1]
        z_next = z(a_i, theta[i+1])
        a_next = activation(z_next)
        activations[i+2] = a_next

def backprop(y, hidden, activations, theta, alpha, reg):
    m = len(y)
    delta = []
    # Calculating local gradients
    # Output layer
    y_pred = activations[-1]
    delta_i = y_pred - y
    delta.append(delta_i)

    start = len(activations) - 1

    # Hidden layers
    for i in range(start, 1, -1): # we don't calculate errors for input layer
        theta_prev = theta[i-1][:,1:] # this is ignoring bias
        tmp = theta_prev.T @ delta_i
        delta_i = tmp * ( activations[i-1][1:] * (1 - activations[i-1][1:])) # this is ignoring bias
        delta.append(delta_i)

    # delta list holds the values
    # delta length is all but first layer (layers-1)
    # we could add an extra column for input layer to have the same
    # indexes as in the slides
    delta.reverse()

    # Calculating Deltas
    num_clases = theta[-1].shape[0]
    Delta = initialize_weights(X, num_clases, hidden) # The Delta has the same dimensions as the weight matrix (we will ignore bias though)
    # This is for weights, not for neurons, that is why we reach layer 1
    start = len(delta) - 1
    for i in range(start, -1, -1):
        activations_this_layer = activations[i][1:,:] # this is ignoring bias
        delta_next_layer = delta[i]
        activations_times_delta = activations_this_layer@delta_next_layer.T
        Delta[i][:,1:] = activations_times_delta.T

    D = [ x/m for x in Delta]

    for i in range(len(D)):
        d = D[i]
        t = theta[i]
        d[:,1:] += reg * t[:,1:]

    for i in range(len(theta)):
        t = theta[i]
        d = D[i]

        t[:, 1:] = t[:, 1:] - (alpha * d[:, 1:])

def get_config_for_example():
    X = np.array([[0.05], [0.10]])
    y = np.array([[0.01], [0.99]])

    hidden = [2]
    # el dos es por las neuronas en la capa de salida
    theta = initialize_weights(X, 2, hidden)

    theta_1 = theta[0]
    theta_1[0,0] = 0.35
    theta_1[0,1] = 0.15
    theta_1[0,2] = 0.20

    theta_1[1,0] = 0.35
    theta_1[1,1] = 0.25
    theta_1[1,2] = 0.30

    theta_2 = theta[1]
    theta_2[0,0] = 0.60
    theta_2[0,1] = 0.40
    theta_2[0,2] = 0.45

    theta_2[1,0] = 0.60
    theta_2[1,1] = 0.50
    theta_2[1,2] = 0.55

    return (X, y, hidden, theta)

def predict(X, theta):
    a_i = X
    # all theta layers
    for i in range(len(theta)):
        biases = np.ones(a_i.shape[1])
        a_i = np.vstack((biases, a_i))

        z_next = z(a_i, theta[i])
        a_i = activation(z_next)

    return a_i

def get_one_hot(targets, nb_classes):
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape)+[nb_classes])

if __name__ == "__main__":
    (X, y, hidden, theta) =  get_config_for_example()
    activations = initialize_activations(X,output_neurons=2, hidden=hidden)

    for i in range(10000):#epoch loop
        forward(X, hidden, activations, theta)
        backprop(y, hidden, activations, theta, alpha=0.5, reg=0)
    y_pred = predict(X, theta)
    print('{} pred as {}, should be {}'.format(X.T, y_pred.T, y.T))
