from dataset_reader import DatasetReader
from nn import NeuralNetwork

if __name__ == "__main__":
  
  # DatasetReader
  data = DatasetReader(name='digits')
  data.split_data(0.20, one_hot=True)
  data.normalize_data('minmax')

  # Specify number input/output nodes
  input_nodes = data.X_train.shape[1]
  output_nodes = data.y_train.shape[1]

  # Specify network structure
  layers = [[input_nodes, 'input'],
            [32, 'relu'],
            [output_nodes,'softmax']]
  
  # Train and plot
  nn = NeuralNetwork(layers=layers)
  nn.train(data.X_train, data.y_train, 
           data.X_test, data.y_test, 
           epochs=30, batch_size=32)
  nn.plot_history(start=1)
