import numpy as np
import neuralNetwork as nn
import dataset as d
import ecoParams as param

class EcosystemAlgorithm:
    def __init__(self, NeuralNetwork):
        self.NeuralNetwork = NeuralNetwork
        self.best_plant = None
        self.best_herbivore = None
        self.best_predator = None
      
    def perform_movement(self, Animals, best_so_far, best_prey):
         for animal in Animals:
            velocity = self.count_velocity(animal, best_so_far, best_prey)
            animal.set_velocity(velocity)
            position = animal.position + animal.velocity
            animal.set_position(position)
     
    def count_velocity(self, Animal, best_animal_so_far, best_prey_so_far):
        r = np.random.rand()
        pt_prev_v = Animal.mcp[0] * r * Animal.velocity
        pt_best = Animal.mcp[1] * r * (Animal.best_organism[1] - Animal.position)
        pt_best_so_far = Animal.mcp[2] * r * (best_animal_so_far[1] - Animal.position)
        if Animal.best_neighbour is not None:
            pt_best_neighbour = Animal.mcp[3] * r * (Animal.best_neighbour[1] - Animal.position)
        else: pt_best_neighbour = 0
        pt_assign_prey = Animal.mcp[4] * r * (Animal.prey.position - Animal.position)
        pt_best_prey = Animal.mcp[5] * r * (best_prey_so_far[1] - Animal.position)
        
        return pt_prev_v + pt_best + pt_best_so_far + pt_best_neighbour + pt_assign_prey + pt_best_prey
        
    def save_best_neighbour(self, Animals):
        for animal in Animals:
            rest = np.array([n for n in Animals if n != animal])
            bottom = animal.position - param.neighbourhood_r
            top = animal.position + param.neighbourhood_r
            neighbours = np.array([n for n in rest 
                            if np.logical_and(bottom < n.position, n.position < top).all()])
            if neighbours.size:
                best_neighbour = min(neighbours, key=lambda neighbours: neighbours.adaptation)
                animal.set_best_neighbour(best_neighbour)

    # TODO make sperate function for seting best animal so far
    def save_bests(self, Plants, Herbivores, Predators):
        for p in Plants: p.save_best_organism()
        for h in Herbivores: h.save_best_organism()
        for p in Predators: p.save_best_organism()
            
        best_plant = self.find_best_organism(Plants)    
        best_herbivore = self.find_best_organism(Herbivores)
        best_predator = self.find_best_organism(Predators)
        
        if self.best_plant is not None:
            if self.best_plant[0] >  best_plant[0]:
                self.best_plant = best_plant
        else: self.best_plant = best_plant
        
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
            
    def find_best_organism(self, Organisms):
        best = min(Organisms, key=lambda Organisms: Organisms.adaptation)

        return best.adaptation, best.position
    
    def create_population(self):
        plants = self.create_plant_population()
        herbivores = self.create_herbivore_population(plants)
        predators = self.create_predator_population(herbivores)
        
        return plants, herbivores, predators
    
    def create_predator_population(self, Herbiovores):
        predators = np.empty(shape=(param.n_pr,), dtype=object)
        
        for i in range(param.n_pr):
            position = self.deploy_position()
            adaptation = self.count_adaptation(position)
            mcp = np.random.uniform(-0.5, 2, 6)
            prey = np.random.choice(Herbiovores)
            predators[i] = Predator(i, position, adaptation, mcp , 0, prey)
            
        return predators
    
    def create_herbivore_population(self, Plants):
        herbivores = np.empty(shape=(param.n_he,), dtype=object)
        
        for i in range(param.n_he):
            position = self.deploy_position()
            adaptation = self.count_adaptation(position)
            mcp = np.random.uniform(-0.5, 2, 6)
            prey = np.random.choice(Plants)
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
        self._best_organism = None

    @property
    def best_organism(self):
        return self._best_organism
    
    def save_best_organism(self):
        if self.best_organism is not None:
            if self.best_organism[0] < self.adaptation:
                self._best_organism = [self.adaptation, self.position]
        else:
            self._best_organism = [self.adaptation, self.position]
    
    def set_adaptation(self, adaptation):
        self.adaptation = adaptation
        
    def set_position(self, position):
        self.position = position

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
        self._best_neighbour = None
    
    @property
    def best_neighbour(self):
        return self._best_neighbour
        
    def set_best_neighbour(self, best_neighbour):
        self._best_neighbour = best_neighbour.adaptation, best_neighbour.position
        
    def set_velocity(self, velocity):
        self.velocity = velocity
        
class Herbivore(Animal):
    def __init__(self, id, position, adaptation, mcp, velocity, prey):
        super().__init__(id, position, adaptation, mcp, velocity, prey)       
    

class Predator(Animal):
    def __init__(self, id, position, adaptation, mcp, velocity, prey):
        super().__init__(id, position, adaptation, mcp, velocity, prey)
        self.life = param.life
        
def main():
    data = d.Dataset()
    input, output = data.get_dataset()
    neural_network = nn.NeuralNetwork(input, output)
    eco = EcosystemAlgorithm(neural_network)

    plants = eco.create_plant_population()
    herbivores = eco.create_herbivore_population(plants)
    predators = eco.create_predator_population(herbivores)

    eco.save_bests(plants, herbivores, predators)

    eco.perform_movement(herbivores, eco.best_herbivore, eco.best_plant)

if __name__ == "__main__":
    main()
