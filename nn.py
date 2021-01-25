import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import log_loss
from dataset_reader import get_mini_batches
from nn_functions import relu, sigmoid, softmax


class NeuralNetwork:
  """
  Class to represent a Neural Network.
  
  Attributes
  ----------
  layers : 2D array
    Specifies number of nodes for each layer and each layer's activation function
  param_dict : dictionary
    Stores weight, bias, z and activation for each layer in the NN
  history : dictionary
    Stores train/val loss and accuracy for each epoch during training
  lr : float
    Learning rate. The step size towards convergence when applying gradient descent

  Methods
  -------
  feed_forward(X_train)
    Feed training data into our network, saving each layers activation and z into the networks param_dict
  back_prop(y)
    Calculate gradient of loss function with respect to weights so we can later apply gradient descent
  gradient_descent()
    Update weight and bias with weight and bias derivatives
  check_accuracy(y, y_pred)
    Calculate accuracy of our model's predictions if output is binary or multiple classes
  train(X_train, y_train, X_val=None, y_val=None, epochs=10, batch_size=None lr=0.001)
    Given the data, train our model for set amount of epochs
  plot_history(start=0, end=None)
    Plot with matplotlib the performance of each epoch when model training is complete
  """

  def __init__(self, layers):
    """
    Parameters
    ----------
    layers : 2D list
      Specify number of nodes and activation function for each layer
    """
    self.layers = layers
    self.param_dict = {}
    self.history = {"Training Loss": [],
                    "Training Accuracy": [],
                    "Validation Loss": [],
                    "Validation Accuracy": []
                    }
    self.lr = 0.001

    # Initialize weight matrices for hidden layers
    for i, layer in enumerate(self.layers):
      if i != 0:
        num_inputs = self.layers[i-1][0]
        num_output = self.layers[i][0]
        
        self.param_dict["Weight"+str(i)] = np.random.randn(num_inputs, num_output)
        self.param_dict["Bias"+str(i)] = np.zeros((1, num_output))
          
  def feed_forward(self, X_train):
    """ Feed training data into our network, saving each layers activation and z into the networks param_dict

    Parameters
    ----------
    X_train : numpy array
      Attribute values of our training data
      
    """
    self.param_dict['Activation0'] = X_train
    prev_activation = self.param_dict['Activation0']
  
    for i, layer in enumerate(self.layers):
      if i != 0:
        cur_weight = self.param_dict["Weight"+str(i)]
        cur_bias = self.param_dict["Bias"+str(i)]
        z = prev_activation.dot(cur_weight) + cur_bias
                
        if self.layers[i][1] == 'sigmoid':
          prev_activation = sigmoid(z)
        elif self.layers[i][1] == 'relu':
          prev_activation = relu(z)
        else:
          prev_activation = softmax(z)
          
        self.param_dict['Activation'+str(i)] = prev_activation
        self.param_dict["Z"+str(i)] = z

  def back_prop(self, y):
    """ Calculate gradient of loss function with respect to weights so we can later apply gradient descent

    Parameters
    ----------
    y : numpy array
      The true target values of each instance in our training data

    """
    i = len(self.layers)-1
    param = self.param_dict
    y_pred = param['Activation'+str(i)]    
    
    # Calculate derivatives for last layer
    param['Z'+str(i)+'_d'] = y_pred - y
    param['Weight' + str(i)+'_d'] = np.dot(param['Activation'+str(i-1)].T, param['Z'+str(i)+'_d']) 
    param['Bias' + str(i)+'_d'] = np.sum(param['Z' + str(i)+'_d'], axis=0, keepdims=True)

    i-=1
    
    # Calculate derivatives for remaining layers
    while i > 0:   
      if self.layers[i][1] == 'sigmoid':
        activation_func = sigmoid
      elif self.layers[i][1] == 'relu':
        activation_func = relu
        
      param['Z'+str(i)+'_d'] = np.dot( param['Z'+str(i+1)+'_d'], param['Weight'+str(i+1)].T) * activation_func(param['Activation'+str(i)], derivative=True)
      param['Weight'+str(i)+'_d'] = np.dot(param['Activation'+str(i-1)].T, param['Z'+str(i)+'_d'] )
      param['Bias'+str(i)+'_d'] = np.sum(param['Z' + str(i)+'_d'], axis=0, keepdims=True) 
      i-=1
      
    self.gradient_descent()
      
  def gradient_descent(self):
    """ Update weight and bias with gradient descent """
    param = self.param_dict
    for i in range(1,len(self.layers)):
      param["Weight"+str(i)] -= self.lr* param['Weight' + str(i)+'_d']
      param["Bias"+str(i)] -= self.lr * param['Bias' + str(i)+'_d']
      
  def check_accuracy(self, y, y_pred):
    """ Calculate accuracy of our model's predictions if output is binary or multiple classes

    Parameters
    ----------
    y : numpy array
      The true target values of each instance in our training data
    y_pred : numpy array
      The model's predicted target values of each instance in our training data
    
    Returns
    -------
    float
      The accuracy of our model given in a probability value between 0 and 1.
    """
    total = 0
    if y.shape[1] == 1:
      for i, p in enumerate(y_pred):
        if p > 0.5 and 1 == y[i]:
          total += 1
        elif p < 0.5 and 0 == y[i]:
          total += 1
    else:
      for i, p in enumerate(y_pred):
        if np.argmax(p) == np.argmax(y[i]):
          total+=1
    
    return total/len(y)
      
  def train(self, X_train, y_train, X_val=None, y_val=None, epochs=10, batch_size=None, lr=0.001):
    """ Given the data, train our model for set amount of epochs. When each epoch completes, train/val loss and accuracy is printed.  Each epoch loss and accuracy is saved in networks history dictionary.  

    Parameters
    ----------
    X_train : numpy array
      Attribute values of our training data
    y_train : numpy array
      The true target values of each instance in our training data
    X_val:
      Attribute values of our testing data. default=None
    y_val : numpy array
      The true target values of each instance in our testing data. default=None
    epochs : int
      How many times do we feed all of our data into the network.
    batch_size : int
      How many instances should we feed into our network until we update our weights with gradient descent.
    lr : float
      Learning rate.  Size of our steps when we apply gradient descent
    
    """
    self.lr = lr
    self.history = {
                "Training Loss" : [],
                "Training Accuracy": [],
                "Validation Loss": [],
                "Validation Accuracy" : [], 
              }
    
    for epoch in range(epochs):
      print("==========================================")
      print("Training Epoch: ", epoch)
      print("==========================================")
      
      if batch_size:
        batchs_x, batches_y = get_mini_batches(X_train, y_train)
      
        for (batch_x, batch_y) in zip(batchs_x, batches_y):
          self.feed_forward(batch_x)
          self.back_prop(batch_y)
      else:
        self.feed_forward(X_train)
        self.back_prop(y_train)

      self.feed_forward(X_train)
      train_predictions = self.param_dict["Activation"+str(len(self.layers)-1)]
      self.history['Training Loss'].append(log_loss(y_train, train_predictions))
      self.history['Training Accuracy'].append(self.check_accuracy(y_train, train_predictions))
      
      self.feed_forward(X_val)
      val_predictions = self.param_dict["Activation"+str(len(self.layers)-1)]
      self.history['Validation Loss'].append(log_loss(y_val, val_predictions))
      self.history['Validation Accuracy'].append(self.check_accuracy(y_val, val_predictions))
        
      for metric, value in self.history.items():
        print("%s: %.4f" % (metric, value[-1]))

  def plot_history(self, start=0, end=None):
    """ Plot the performance of each epoch when model training is complete

    Parameters
    ----------
    start : int
      Epoch to start plotting history from
    end : int
      Which epoch to stop plotting history (default=None)
    
    Returns
    -------

    A Matplotlib plot of each epochs train/val performance
    """
    if not end:
      end = len(self.history['Training Loss'])
      
    loss = self.history['Training Loss'][start:end]
    val_loss = self.history['Validation Loss'][start:end]
    
    accuracy = self.history['Training Accuracy'][start:end]
    val_accuracy = self.history['Validation Accuracy'][start:end]
    epochs = [i for i in range(start, end, 1)]

    plot(loss, val_loss, epochs, "Loss", "Train/Val Loss")
    plot(accuracy, val_accuracy, epochs, "Accuracy", "Train/Val Accuracy")

def plot(train, val, epochs, label, title):
  """ Boiler plate code for generating and displaying plots with matplotlib

  Parameters
  ----------
  train : int
    Performance measurments of the training set
  val : int
    Performance measurments of the validation set
  epochs : int
    Number of epochs for training
  label : string
    Specify what measurment we are plotting
  title : string
    Specifies title of plot

  """
  plt.plot(epochs, train, 'bo', label='Training '+str(label))
  plt.plot(epochs, val, 'r', label='Validation '+str(label))
  
  plt.title(title)
  plt.xlabel('Epochs')
  plt.ylabel(label)
  plt.legend()
  plt.show()

   
    

