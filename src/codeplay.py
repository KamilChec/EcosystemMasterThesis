import numpy as np
import dataset as d
import neuralNetwork as nn

data = d.Dataset()
input, output = data.get_dataset()
neural_network = nn.NeuralNetwork(input, output)
network_parameters = neural_network.create_population(3)
# for p in network_parameters: position = np.append(position, np.ravel(p)) 
position = []
position = [[p for p in np.ravel(parametr)] for parametr in network_parameters]

print(position)