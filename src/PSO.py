import numpy as np
from numpy.lib.function_base import diff
import neuralNetwork as nn
import dataset as d
import matplotlib.pyplot as plt

class ParticleSwarmOptimization:
    nn_hidm = 3
    n_particles = 20
    
    c1 = c2 = 0.1
    w = 0.8
    
    gbest = 999
    gbest_score = 999
    
    all_generation_adaptations = []
    
    def __init__(self, num_population, w, c1, c2, neuralNetwork):
        self.num_population = num_population
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.neuralNetwork = neuralNetwork
        
    def create_population(self):
        population = []
        for i in range(self.num_population):
            position = self.neuralNetwork.create_population(self.nn_hidm)
            velocity = [p * 0.1 for p in self.neuralNetwork.create_population(self.nn_hidm)]
            population.append(Specimen(i, position, velocity, score=0))

        return population
    
    def set_score(self, population):
        for specimen in population:
            self.neuralNetwork.set_network_parameters(*specimen.position)
            specimen.set_score(self.neuralNetwork.calculate_loss())
            
    def set_pbest(self, population):
        for specimen in population:
            if specimen.pbest_score > specimen.score:
                specimen.set_pbest(specimen.position)
                specimen.set_pbest_score(specimen.score)
        
    def set_gbest(self, population):
        for specimen in population:
            if specimen.pbest_score < self.gbest_score:
                self.gbest = specimen.pbest
                self.gbest_score = specimen.pbest_score
                
    def set_velocity(self, population):
        for specimen in population:
            r = np.random.rand(2)
            V1 = self.mk_multiplication(specimen.velocity, self.w)
            V2 = self.mk_multiplication(self.mk_subtraction(specimen.pbest, specimen.position), self.c1 * r[0])
            V3 = self.mk_multiplication(self.mk_subtraction(self.gbest, specimen.position), self.c2 * r[1])
            V = []
            for v1, v2, v3 in zip(V1, V2, V3):
                V.append(v1 + v2 + v3)
            specimen.set_velocity(V)
    
    def perform_movement(self, population):
        for specimen in population:
            a = self.mk_addition(specimen.position, specimen.velocity)
            specimen.set_position(a)
            
    def mk_subtraction(self, arr1, arr2):
        difference = []
        for el1, el2 in zip(arr1, arr2):
            difference.append(el1 - el2)
        
        return difference
    
    def mk_addition(self, arr1, arr2):
        total = []
        for el1, el2 in zip(arr1, arr2):
            total.append(el1 + el2)
        
        return total
    
    def mk_multiplication(self, arr, cons):
        product = []
        for el in arr:
            product.append(el * cons)
        
        return product
    
    def set_all_parameters(self, population):
        self.set_score(population)
        self.set_pbest(population)
        self.set_gbest(population)
        self.set_velocity(population)
        
    def print_loss(self, iteration, first_use=False):
        self.all_generation_adaptations.append(self.gbest_score)
        mode = 'w' if first_use else 'a'
        with open('res/psoData.txt', mode) as file:
            file.write("Loss after iteration {}:\t{}\n".format(iteration, self.gbest_score))
            
    def plot_evolution_of_adaptation(self):
        nn_adaptations = len(self.all_generation_adaptations)
        x = np.linspace(0, nn_adaptations, nn_adaptations)
        plt.plot(x, self.all_generation_adaptations, '-o')
        plt.show()
        
class Specimen:
    pbest = 9999
    pbest_score = 9999
    
    def __init__(self, id, position, velocity, score):
        self.id = id
        self.position = position
        self.velocity = velocity
        self.score = score
        
    def set_position(self, position):
        self.position = position
        
    def set_velocity(self, velocity):
        self.velocity = velocity
        
    def set_score(self, score):
        self.score = score
    
    def set_pbest(self, pbest):
        self.pbest = pbest
    
    def set_pbest_score(self, pbest_score):
        self.pbest_score = pbest_score
        
    def __str__(self):
        return 'Specimen numb.: ' + str(self.id) + '\n ' + str(self.position) + '\nscore: ' + str(self.score) + '\n'        
        
# data = d.Dataset()
# input, output = data.get_dataset()
# neural_network = nn.NeuralNetwork(input, output)
# pso = ParticleSwarmOptimization(100, 0.8, 1, 0, neural_network)

# population = pso.create_population()
    
# for i in range(100):
#     pso.set_all_parameters(population)
#     pso.perform_movement(population)
#     if i % 10 == 0:
#         if (i == 0): 
#             pso.print_loss(i, first_use=True)
#         else:
#             pso.print_loss(i)
#         if pso.c1 > 0.1: pso.c1 -= 0.1
#         pso.c2 += 0.1
        
# pso.plot_evolution_of_adaptation()