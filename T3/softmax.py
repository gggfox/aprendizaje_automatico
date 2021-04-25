import math
import numpy as np 

layer_outputs = [[4.8,1.21,2.385],[8.9,-1.81,0.2],[1.41,1.051,0.026]]
#2.718
E = math.e


exp_values = np.exp(layer_outputs)

#normalize
norm_values = exp_values / np.sum(exp_values, axis=1, keepdims=True)#axis 1 is the sum of each row

print(norm_values)