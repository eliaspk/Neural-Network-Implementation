# Neural Network Implementation

## About the project
Developed a multi layer neural network using numpy. Its functionality is currently restricted to classifcation problems. It's currently set up to be trained on MNIST, iris, and breast_cancer which are loaded through sklearn. Prints the train/validation of each epoch in terminal, and history plots are also displayed after training.

## Usage
Install necessary packages through
```sh
$ pip install -r requirements.txt
```
Once in the project directory, the project can be run with
```sh
$ python main.py
```
There are some areas within `main.py` that can be played around with.

`DatasetReader` takes in parameters `digits` `iris` or `breast_cancer`  
`normalize_data` takes in parameters `minmax` or `standard`
```python
data = DatasetReader(name='digits')
data.normalize_data('minmax')
```
In `layers` additional layers can be added in the format `[num_nodes, activation]`
Current activation functions supported are `relu` `sigmoid` and `softmax`
```python
layers = [[input_nodes, 'input'],
        [32, 'relu'],
        [output_nodes,'softmax']]
```
