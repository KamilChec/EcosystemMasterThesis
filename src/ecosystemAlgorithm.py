import numpy as np
import neuralNetwork as nn

class EcosystemAlgorithm:
    def __init__(self, NeuralNetwork, n_ppl):
        self.NeuralNetwork = NeuralNetwork
        self.n_ppl = n_ppl
    
    def interaction_plant_herbivore(self):
        pass
    
    def interaction_herbivore_predator(self):
        pass
    
    def crossover_plant(self):
        pass
    
    def crossover_animal(self):
        pass
    
    def create_plant_population(self):  
        plants = []
        
        for i in range(self.n_ppl):
            network_parameters = self.NeuralNetwork.create_population(3)
            position = [np.ravel(p) for p in network_parameters]
            plant = Plant(i, position, 0, 0)
            adaptation = self.count_adaptation(plant)
            plant.set_adaptation(adaptation)
            plants.append(plant)
        
        return plants
            
    def count_plants_size(self, plants):
        sum_adaptation = sum(p.adaptation for p in plants)
        p.set
        
            
            
    def count_adaptation(self, Organism):
        self.NeuralNetwork.set_network_parameters(*Organism.organism_to_neuron())
        
        return self.neural_network.calculate_loss()
            

class Organism: 
    def __init__(self, id, position, adaptation):
        self.id = id
        self.position = position
        self.adaptation = adaptation
        
    def set_adaptation(self, adaptation):
        self.adaptation = adaptation
        
    def create_population(self):
        pass
    
    def calculate_adaptation(self):
        pass
    
    def find_best_organism(self):
        pass
    
    def mutation(self):
        pass
    
    
    def organism_to_neuron(self):
        nn_input_dim = self.nerual_network.nn_input_dim
        nn_output_dim = self.nerual_network.nn_output_dim
        nn_hdim = self.nerual_network.nn_hdim
        
        W1 = self.position[:nn_input_dim*nn_hdim].reshape(nn_input_dim, nn_hdim)
        rest_of_organism = self.position[nn_input_dim*nn_hdim:]
        b1 = rest_of_organism[:nn_hdim]
        rest_of_organism = rest_of_organism[nn_hdim:]
        W2 = rest_of_organism[:nn_output_dim*nn_hdim].reshape(nn_hdim, nn_output_dim)
        rest_of_organism = rest_of_organism[nn_output_dim*nn_hdim:]
        b2 = rest_of_organism[:nn_output_dim]
        
        return W1, b1, W2, b2

class Plant(Organism):
    def __init__(self, id, position, adaptation, size):
        super().__init__(id, position, adaptation)
        self.size = size
        
    def set_size(self, size):
        self.size = size
        
class Animal(Organism):
    def __init__(self, id, position, adaptation, mcp, best_location, velocity, prey):
        super().__init__(id, position, adaptation)
        self.mcp = mcp
        self.best_location = best_location
        self.velocity = velocity
        self.prey = prey
        
    def find_best_location(self):
        pass
    def perform_movement(self):
        pass
    
class Herbivore(Animal):
    def __init__(self, id, position, adaptation, mcp, best_location, velocity, prey):
        super().__init__(id, position, adaptation, mcp, best_location, velocity, prey)
        
    

class Predator(Animal):
    def __init__(self, id, position, mcp, best_location, velocity, prey, life):
        super().__init__(id, position, mcp, best_location, velocity, prey)
        self.life = life