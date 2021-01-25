import numpy as np
import pandas as pd

from sklearn.datasets import load_digits, load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class DatasetReader:
  """
  Class for pre-processing sklearn datasets for neural network training
  
  Attributes
  ----------
  dataset : sklearn dataset object
    Dictionary-like object with several attributes: data, target, feature_names, target_names
  X_train, X_test, y_train, y_test : numpy arrays
    Arrays returned when using sklearn's train_test_split function on dataset

  Methods
  -------
  split_data(test_size, one_hot=False)
    Applies one hot encoding if set to true.  Uses sklearn train_test_split to split data.Reshapes y values to be compatible with training
  normalize_data(scaler)
    Normalizes data by scaler specification. Either StandardScaler or MinMax.

  """
  def __init__(self, name):
    """
    Parameters
    ----------
    name : string
      Specify name of sklearn dataset (digits, iris, breast_cancer)
    """
    load_dataset_func = eval("load_"+str(name))
    self.dataset = load_dataset_func()
    
    self.X_train = None
    self.X_test = None
    self.y_train = None
    self.y_test = None 
    
  def split_data(self, test_size, one_hot=False):
    """ Split data into train and test.  Shape y values to ensure compatability with training

    Parameters
    ----------
    test_size : int
      Percentage of data that will be allocated for testing
    one_hot : boolean
      Specify if y values should be one_hot encoded
    """
    if one_hot:
      target = pd.get_dummies(self.dataset.target)
      y_columns = target.shape[1]
    else:
      y_columns = 1
      target = self.dataset.target

    self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.dataset.data, target, test_size=test_size)
    
    self.y_train = np.reshape(np.array(self.y_train), (self.y_train.shape[0], y_columns ))
    self.y_test = np.reshape(np.array(self.y_test), (self.y_test.shape[0], y_columns ))
  
  def normalize_data(self, scaler):
    """ Normalize data 

    Parameters
    ----------
    scaler : string
      Specifies what method to use for normalizing data
    """
    self.scaler = scaler
    if scaler == 'standard':
      scaler = StandardScaler()
    elif scaler == 'minmax':
      scaler = MinMaxScaler()
    
    self.X_train = scaler.fit_transform(self.X_train)
    self.X_test = scaler.transform(self.X_test)
    
    return self.X_train, self.X_test 

def get_mini_batches(X_train, y_train, batch_size=32):
  """ Creates mini batches for data

  Parameters
  ----------
  X_train : numpy array
    Attribute data
  y_train : numpy array
    Target data
  batch_size : int
    Size for each batch
  """
  batches_x = []
  batches_y = []
  
  indices = np.arange(X_train.shape[0])
  np.random.shuffle(indices)
  
  x = X_train[indices]
  y = y_train[indices]

  for i in range(0, x.shape[0], batch_size ):
    batches_x.append(x[i:i + batch_size ])
    batches_y.append(y[i:i + batch_size ])

  return batches_x, batches_y
  
  
  
  
  
  
  
  