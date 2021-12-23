import numpy as np
import neuralNetwork as nn
import dataset as d

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
            position = self.deploy_position()
            adaptation = self.count_adaptation(position)
            plant = Plant(i, position, adaptation, 0)
            plants.append(plant)
        
        for p in plants: p.set_size(self.count_plants_size(p, plants))
        
        return plants
    
    def deploy_position(self):
            network_parameters = self.NeuralNetwork.create_population(3)
            position = []
            for p in network_parameters: position = np.append(position, np.ravel(p))
            
            return position
            
    def count_plants_size(self, plant, plants):
        sum_adaptation = sum(p.adaptation for p in plants)
        size = plant.adaptation / sum_adaptation * self.n_ppl
        
        return 1/size
        
    def count_adaptation(self, position):
        self.NeuralNetwork.set_network_parameters(*self.organism_to_neuron(position))
        
        return self.NeuralNetwork.calculate_loss()
    
    def organism_to_neuron(self, position):
        nn_input_dim = self.NeuralNetwork.nn_input_dim
        nn_output_dim = self.NeuralNetwork.nn_output_dim
        nn_hdim = self.NeuralNetwork.nn_hdim
        
        W1 = position[:nn_input_dim*nn_hdim].reshape(nn_input_dim, nn_hdim)
        rest_of_organism = position[nn_input_dim*nn_hdim:]
        b1 = rest_of_organism[:nn_hdim]
        rest_of_organism = rest_of_organism[nn_hdim:]
        W2 = rest_of_organism[:nn_output_dim*nn_hdim].reshape(nn_hdim, nn_output_dim)
        rest_of_organism = rest_of_organism[nn_output_dim*nn_hdim:]
        b2 = rest_of_organism[:nn_output_dim]
        
        return W1, b1, W2, b2
            

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
        

data = d.Dataset()
input, output = data.get_dataset()
neural_network = nn.NeuralNetwork(input, output)
eco = EcosystemAlgorithm(neural_network, 3)

plants = eco.create_plant_population()

for p in plants: print(p.size)