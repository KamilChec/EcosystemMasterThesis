import neuralNetwork as nn
import dataset as d
import numpy as np

class GeneticAlgorithm:
    nn_hidm = 3
    
    def __init__(self, num_population, neuralNetwork):
        self.num_population = num_population
        self.neuralNetwork = neuralNetwork
    
    def create_population(self):
        chromosomes = []
        for i in range(self.num_population):
            chromosome = []
            network_parameters = self.neuralNetwork.create_population(self.nn_hidm)
            for parameter in network_parameters:
                chromosome = np.append(chromosome, np.ravel(parameter))
            chromosomes.append(Chromosome(i, chromosome, adaptation=0))
        return chromosomes
    
    def chromosome_to_neuron(self, chromosome):
        nn_input_dim = self.neuralNetwork.nn_input_dim
        nn_hdim = self.nn_hidm
        nn_output_dim = self.neuralNetwork.nn_output_dim
        
        W1 = chromosome.body[:nn_input_dim*nn_hdim].reshape(nn_input_dim, nn_hdim)
        rest_of_chromosome = chromosome.body[nn_input_dim*nn_hdim:]
        b1 = rest_of_chromosome[:nn_hdim]
        rest_of_chromosome = rest_of_chromosome[nn_hdim:]
        W2 = rest_of_chromosome[:nn_output_dim*nn_hdim].reshape(nn_hdim, nn_output_dim)
        rest_of_chromosome = rest_of_chromosome[nn_output_dim*nn_hdim:]
        b2 = rest_of_chromosome[:nn_output_dim]
        
        return W1, b1, W2, b2
    
    def adaptation(self, chromosome):
        self.neuralNetwork.set_network_parameters(*self.chromosome_to_neuron(chromosome))
        loss = self.neuralNetwork.calculate_loss()
        
        return loss
    
    def probability(self):
        pass
    
    def choose_parents(self):
        pass
    
    def selection(self):
        pass
    
    def crossover(self):
        pass
    
    def mutation(self):
        pass
    
    def best_chromosome(self):
        pass
    
class Chromosome:
    def __init__(self, id, body, adaptation):
        self.id = id
        self.body = body
        self.adaptation = adaptation
    
    def set_adaptation(self, adaptation):
        self.adaptation = adaptation
    
    def set_body(self, body):
        self.body = body
        
    def __str__(self):
        return 'Chromosome numb.: ' + str(self.id) + '\n ' + str(self.body) + '\nadaptation: ' + str(self.adaptation) + '\n'

data = d.Dataset()
input, output = data.get_dataset()        
neural_network = nn.NeuralNetwork(input, output)
ga_algorithm = GeneticAlgorithm(1, neural_network)

chromosomes = ga_algorithm.create_population()

print(ga_algorithm.adaptation(chromosomes[0]))
