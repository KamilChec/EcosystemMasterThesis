from operator import attrgetter
import neuralNetwork as nn
import dataset as d
import numpy as np
import random
import matplotlib.pyplot as plt
import math

class GeneticAlgorithm:
    nn_hidm = 3
    all_generation_adaptations = []
    
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
    
    def set_adaptation(self, chromosomes):
        for chromosome in chromosomes:
            chromosome.set_adaptation(self.adaptation(chromosome))

    def probability_of_selecting(self, chromosomes):
        log_adaptation_sum = 0
        log_adaptation_sum = sum([math.exp(-ch.adaptation*2) for ch in chromosomes])
        probabilities = [math.exp(-ch.adaptation*2)/log_adaptation_sum for ch in chromosomes]
        norm_factor = 1 / sum(probabilities) 
        normalised = [norm_factor * p for p in probabilities] 
        
        return normalised
    
    def select_parents(self, chromosomes):
        choice = np.random.choice(chromosomes, self.num_population, p=self.probability_of_selecting(chromosomes))
        
        return choice
    
    def crossover(self, chromosomes):
        offspring = []
        parents = self.select_parents(chromosomes)
        available_parents = list(range(self.num_population))
        for i in range(self.num_population):
            if i % 2 == 0:
                crossing_point = np.random.randint(len(chromosomes[0].body))
            else:
                mother_idx = random.choice(available_parents)
                available_parents.remove(mother_idx)
                father_idx = random.choice(available_parents)
                available_parents.remove(father_idx)
                mother, father = parents[mother_idx].body.copy(), parents[father_idx].body.copy()
                
                buff = father[crossing_point:].copy()
                father[crossing_point:] = mother[crossing_point:].copy()
                mother[crossing_point:] = buff.copy()
                offspring.append(father.copy())
                offspring.append(mother.copy())
                
        return offspring
    
    def mutation(self, offspring):
        for chromosome in offspring:
            if random.random() < 0.03:
                mutation_point = random.randint(0, len(offspring[0]) - 1)
                chromosome[mutation_point] = random.uniform(-5, 5)
                  
        return offspring
            
    def set_new_population(self, offspring, chromosomes):
        for i in range(len(offspring)):
            chromosomes[i].set_body(offspring[i])
            chromosomes[i].set_adaptation(self.adaptation(chromosomes[i]))
    
    def best_chromosome(self, chromosomes):
        return min(chromosomes, key=attrgetter('adaptation'))
    
    def print_loss(self, chromosomes, iteration, first_use=False):
        adaptation = self.best_chromosome(chromosomes).adaptation
        self.all_generation_adaptations.append(adaptation)
        mode = 'w' if first_use else 'a'
        with open('res/gaData.txt', mode) as file:
                file.write("Loss after iteration {}:\t{}\n".format(iteration, adaptation))
    
    def plot_evolution_of_adaptation(self):
        nn_adaptations = len(self.all_generation_adaptations)
        x = np.linspace(0, nn_adaptations, nn_adaptations)
        plt.plot(x, self.all_generation_adaptations, '-o')
        plt.show()
    
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


# data = d.Dataset()
# input, output = data.get_dataset()
# neural_network = nn.NeuralNetwork(input, output)
# genetic_algorithm = GeneticAlgorithm(20, neural_network)
# chromosomes = genetic_algorithm.create_population()


# genetic_algorithm.set_adaptation(chromosomes)

# for chrom in chromosomes:
#     print(chrom.adaptation)

# parents = genetic_algorithm.select_parents(chromosomes)
# print('\n')
# for chrom in parents:
#     print(chrom.adaptation)
    

