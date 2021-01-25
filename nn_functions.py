import numpy as np

def relu(z, derivative=False):
  if derivative:
    z[z<=0] = 0
    z[z>0] = 1
  return z * (z > 0)

def sigmoid(z, derivative=False):
  result = 1 / (1 + np.exp(-z))
  if derivative:
    return result * (1 - result)
  return result

def softmax(z):
  exps = np.exp(z - np.max(z, axis=1, keepdims=True))
  return exps/np.sum(exps, axis=1, keepdims=True)

  