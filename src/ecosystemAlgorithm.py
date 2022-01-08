import numpy as np
import neuralNetwork as nn
import dataset as d
import ecoParams as param

class EcosystemAlgorithm:
    def __init__(self, NeuralNetwork):
        self.NeuralNetwork = NeuralNetwork
        self.best_herbivore = None
        self.best_predator = None
    
    def count_velocity(self, Animal, best_animal_so_far):
        r = np.random.rand()
        pt_prev_v = Animal.mcp[0] * r * Animal.velocity
        pt_best = Animal.mcp[1] * r * Animal.best_animal[1] - Animal.position
        pt_best_so_far = Animal[1] * r * best_animal_so_far[1] - Animal.position
        pass

    def save_best_neighbour(self, Animals):
        for animal in Animals:
            rest = np.array([n for n in Animals if n != animal])
            bottom = animal.position - param.neighbourhood_r
            top = animal.position + param.neighbourhood_r
            neighbours = np.array([n for n in rest 
                            if np.logical_and(bottom < n.position, n.position < top).all()])
            if neighbours.size: 
                animal.set_best_neighbour(min(neighbours, key=lambda neighbours: neighbours.adaptation))            

    def save_bests(self, Herbivores, Predators):
        for herbiovore, predator in zip(Herbivores, Predators):
            herbiovore.save_best_animal()
            predator.save_best_animal()
            
        best_herbivore = self.find_best_animal(Herbivores)
        best_predator = self.find_best_animal(Predators)
        
        if self.best_herbivore is not None:
            if self.best_herbivore[0] >  best_herbivore[0]:
                self.best_herbivore = best_herbivore
        else: self.best_herbivore = best_herbivore
            
        if self.best_predator is not None:
            if self.best_predator[0] >  best_predator[0]:
                self.best_predator = best_predator
        else: self.best_predator = best_predator

        self.save_best_neighbour(Herbivores)
        self.save_best_neighbour(Predators)
            
    def find_best_animal(self, Animals):
        best = min(Animals, key=lambda Animal: Animal.adaptation)
        
        return best.adaptation, best.position
    
    def create_population(self):
        plants = self.create_plant_population()
        herbivores = self.create_herbivore_population()
        predators = self.create_predator_population()
        
        return plants, herbivores, predators
    
    def create_predator_population(self):
        predators = np.empty(shape=(param.n_pr,), dtype=object)
        
        for i in range(param.n_pr):
            position = self.deploy_position()
            adaptation = self.count_adaptation(position)
            mcp = np.random.uniform(-0.5, 2, 6)
            prey = np.random.randint(0, param.n_he) 
            predators[i] = Predator(i, position, adaptation, mcp , 0, prey)
            
        return predators
    
    def create_herbivore_population(self):
        herbivores = np.empty(shape=(param.n_he,), dtype=object)
        
        for i in range(param.n_he):
            position = self.deploy_position()
            adaptation = self.count_adaptation(position)
            mcp = np.random.uniform(-0.5, 2, 6)
            prey = np.random.randint(0, param.n_pl)
            herbivores[i] = Herbivore(i, position, adaptation, mcp, 0, prey)
            
        return herbivores
    
    def create_plant_population(self):  
        plants = np.empty(shape=(param.n_pl,), dtype=object)
        
        for i in range(param.n_pl):
            position = self.deploy_position()
            adaptation = self.count_adaptation(position)
            plants[i] = Plant(i, position, adaptation)
        
        for p in plants: p.set_size(self.count_plants_size(p, plants))
        
        return plants
    
    def deploy_position(self):
            network_parameters = self.NeuralNetwork.create_population(3)
            position = []
            for p in network_parameters: position = np.append(position, np.ravel(p))
            
            return position
            
    def count_plants_size(self, plant, plants):
        sum_adaptation = sum(p.adaptation for p in plants)
        size = plant.adaptation / sum_adaptation * param.n_pl
        
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
    def __init__(self, id, position, adaptation):
        super().__init__(id, position, adaptation)
        self._size = None
        
    @property
    def size(self):
        return self._size
        
    def set_size(self, size):
        self._size = size
        
class Animal(Organism):
    def __init__(self, id, position, adaptation, mcp, velocity, prey):
        super().__init__(id, position, adaptation)
        self.velocity = velocity
        self.prey = prey
        self.mcp = mcp
        self._best_animal = None
        self._best_neighbour = None
    
    @property
    def best_animal(self):
        return self._best_animal

    @property
    def best_neighbour(self):
        return self._best_neighbour
        
    def save_best_animal(self):
        if self.best_animal is not None:
            if self.best_animal[0] < self.adaptation:
                self._best_animal = [self.adaptation, self.position]
        else:
            self._best_animal = [self.adaptation, self.position]
    
    def set_best_neighbour(self, best_neighbour):
        self._best_neighbour = best_neighbour
            
        
class Herbivore(Animal):
    def __init__(self, id, position, adaptation, mcp, velocity, prey):
        super().__init__(id, position, adaptation, mcp, velocity, prey)       
    

class Predator(Animal):
    def __init__(self, id, position, adaptation, mcp, velocity, prey):
        super().__init__(id, position, adaptation, mcp, velocity, prey)
        self.life = param.life
        

data = d.Dataset()
input, output = data.get_dataset()
neural_network = nn.NeuralNetwork(input, output)
eco = EcosystemAlgorithm(neural_network)

plants = eco.create_plant_population()
herbivores = eco.create_herbivore_population()
predators = eco.create_predator_population()


eco.save_bests(herbivores, predators)

print(herbivores[0].best_neighbour.adaptation)
# print(predators[0].best_neighbour)