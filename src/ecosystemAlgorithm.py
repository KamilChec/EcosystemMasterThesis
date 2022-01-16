import numpy as np
import neuralNetwork as nn
import dataset as d
import ecoParams as param
import sys
import math
from tqdm import tqdm

class EcosystemAlgorithm:
    #TODO merge plant_progress with best_plan and so on
    def __init__(self, NeuralNetwork):
        self.NeuralNetwork = NeuralNetwork
        self.best_plant = None
        self.best_herbivore = None
        self.best_predator = None
        self.plant_progress = []
        self.herbivore_progress = []
        self.predator_progress = []
        
        self.deaths = 0
        self.mutations = 0
        
        self.mcp_explore = 1.5
        self.mcp_exploitation = 0
        
    def exploration_exploitation_change(self):
        if self.mcp_explore > 0.1: self.mcp_explore -= 0.1
        if self.mcp_exploitation < 1.5: self.mcp_exploitation += 0.1
        
     #TODO zrobić wypisanie na sam koniec z tablicy   
    def print_loss(self, Plants, Herbivores, Predators, iteration, first_use=False):
        best_plant = self.find_best_organism(Plants)[0]    
        best_herbivore = self.find_best_organism(Herbivores)[0]
        best_predator = self.find_best_organism(Predators)[0]
        
        self.plant_progress.append(best_plant)
        self.herbivore_progress.append(best_herbivore)
        self.predator_progress.append(best_predator)
        
        mode = 'w' if first_use else 'a'
        with open('res/ecoData.txt', mode) as file:
                file.write(
                    "Loss after iteration {} :\tplants: {}\therbivores: {}\tpredators: {}\n".format(
                        iteration, best_plant, best_herbivore, best_predator))
        
    def animal_mutation(self, Animals, mutation_param):
        for animal in Animals:
            if np.random.random() < mutation_param:
                self.mutations += 1
                mutation_point = np.random.randint(animal.mcp.size - 1)
                animal.mcp[mutation_point] = np.random.uniform(
                    param.mcp_scope[0], param.mcp_scope[1])
        
        return Animals
                  
    def plant_mutation(self, Plants):
        for plant in Plants:
            if np.random.random() < param.mutation_pl:
                self.mutations += 1
                mutation_point = np.random.randint(plant.position.size - 1)
                plant.position[mutation_point] = np.random.uniform(-5, 5)
                  
        return Plants
        
    def animal_crossover(self, Object, Animals, Preys, n):
        if Animals.size < n:
            offspring = self.crossover(Animals, n)
            for ii, animal in enumerate(offspring):
                adaptation = self.count_adaptation(animal[0])
                mcp = animal[1]
                prey = np.random.choice(Preys)
                offspring[ii] = Object(animal[0], adaptation, mcp, 0, prey)
                
            return np.concatenate((Animals, offspring))
        else:
            return Animals
        
    def plant_crossover(self, Plants):
        if Plants.size < param.n_pl:
            offspring = self.crossover(Plants, param.n_pl)
            for ii, plant in enumerate(offspring):
                adaptation = self.count_adaptation(plant)
                offspring[ii] = Plant(plant, adaptation)
                
            for p in offspring: p.set_size(self.count_plants_size(p, offspring))

            return np.concatenate((Plants, offspring))
        else:
            return Plants
     
    def crossover(self, Organisms, n):
        refills_numb = n - Organisms.size
        offspring = np.empty(refills_numb, dtype=object)
        probability = self.probability_of_selecting(Organisms)
        for i in range(refills_numb):
            crossing_point = np.random.randint(Organisms[0].position.size)
            mother = np.random.choice(Organisms, p=probability)
            father = np.random.choice(Organisms, p=probability)
                            
            position = np.concatenate(  
                (father.position[:crossing_point], mother.position[crossing_point:]))
            if hasattr(mother, 'mcp'):
                crossing_point = np.random.randint(Organisms[0].mcp.size)
                mcp =  np.concatenate(
                    (father.mcp[:crossing_point], mother.mcp[crossing_point:]))
                offspring[i] = [position, mcp]
            else: offspring[i] = position
                                    
        return offspring             
                
    def probability_of_selecting(self, Organisms):
        log_adaptation_sum = 0
        log_adaptation_sum = sum([math.exp(-o.adaptation*2) for o in Organisms])
        probabilities = [math.exp(-o.adaptation*2)/log_adaptation_sum for o in Organisms]
        norm_factor = 1 / sum(probabilities) 
        normalised = [norm_factor * p for p in probabilities] 
        
        return normalised
            
        
    def herbivore_predator_simulation(self, Herbivores, Predators):
        for predator in Predators:
            if predator.adaptation < predator.prey.adaptation:
                Herbivores = self.dies(predator.prey, Herbivores)
                predator.set_life(predator.life + 1)
                predator.set_prey(np.random.choice(Herbivores))
            else:
                self.run(predator.prey)
                predator.set_position(predator.prey.position)
                predator.set_life(predator.life - 1)
                if predator.life < 0:
                    Predators = self.dies(predator, Predators)
                        
        return Herbivores, Predators
    
    def run(self, runaway):
        v = np.random.uniform(-0.1, 0.1, size=runaway.position.size)
        runaway.set_position(runaway.position + v)
    
    def plant_herbivore_simulation(self, Plants, Herbivores):
        for herbivore in Herbivores:
            try:
                Plants = self.eat_plants(herbivore, Plants, param.nutrition_req)
            except ValueError as err:
                sys.exit(err.args)
        
        return Plants
        
    def eat_plants(self, herbivore, Plants, food_req):
        effect = herbivore.prey.size - food_req
        
        if effect <= 0:
            survived = self.dies(herbivore.prey, Plants)
            if survived.size:
                herbivore.set_prey(np.random.choice(survived))
                return self.eat_plants(herbivore, survived, np.absolute(effect))
            else: raise ValueError('All the plants are eaten and the herbivores want more')
        else: 
            herbivore.prey.set_size(effect)
            
        return Plants
        
    def dies(self, specimen, Organisms):
        self.deaths += 1
        survived = np.array([o for o in Organisms if o != specimen])
        del specimen
        
        return survived
            
    
    def perform_movement(self, Animals, best_so_far, best_prey):
         for animal in Animals:
            # velocity = self.count_velocity(animal, best_so_far, best_prey)
            velocity = self.count_velocity_PSO(animal, best_so_far, best_prey)
            animal.set_velocity(velocity)
            position = animal.position + animal.velocity
            animal.set_position(position)
            
    def count_velocity(self, Animal, best_animal_so_far, best_prey_so_far):
        r = np.random.uniform(0, 0.2)
        pt_prev_v = Animal.mcp[0] * r * Animal.velocity
        pt_best = Animal.mcp[1] * r * (Animal.best_organism[1] - Animal.position)
        pt_best_so_far = Animal.mcp[2] * r * (best_animal_so_far[1] - Animal.position)
        if Animal.best_neighbour is not None:
            pt_best_neighbour = Animal.mcp[3] * r * (Animal.best_neighbour[1] - Animal.position)
        else: pt_best_neighbour = 0
        pt_assign_prey = Animal.mcp[4] * r * (Animal.prey.position - Animal.position)
        pt_best_prey = Animal.mcp[5] * r * (best_prey_so_far[1] - Animal.position)
                 
        return pt_prev_v + pt_best + pt_best_so_far + pt_best_neighbour + pt_assign_prey + pt_best_prey
    
    def count_velocity_PSO(self, Animal, best_animal_so_far, best_prey_so_far):
        r = np.random.rand(6)
        w = 0.8
        pt_prev_v = w * Animal.velocity
        pt_best = self.mcp_explore * r[0] * (Animal.best_organism[1] - Animal.position)
        pt_best_so_far = self.mcp_exploitation * r[1] * (best_animal_so_far[1] - Animal.position)
        if Animal.best_neighbour is not None:
            pt_best_neighbour = 0.3 * r[2] * (Animal.best_neighbour[1] - Animal.position)
        else: pt_best_neighbour = 0
        pt_assign_prey =  self.mcp_explore * r[3] * (Animal.prey.position - Animal.position)
        pt_best_prey =  self.mcp_explore * r[4] * (best_prey_so_far[1] - Animal.position)
        
        return pt_prev_v + pt_best + pt_best_so_far + pt_best_prey + pt_assign_prey + pt_best_neighbour
    
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
    
    def set_adaptation_for_all(self, Plants, Herbivores, Predators):
        self.set_adaptation(Plants)
        self.set_adaptation(Herbivores)
        self.set_adaptation(Predators)
    
    def set_adaptation(self, Organisms):
        for organism in Organisms:
            adaptaion = self.count_adaptation(organism.position)
            organism.set_adaptation(adaptaion)
    
    #TODO concat predator and herbivore population creation
    def create_predator_population(self, Herbiovores):
        predators = np.empty(shape=(param.n_pr,), dtype=object)
        
        for i in range(param.n_pr):
            position = self.deploy_position()
            adaptation = self.count_adaptation(position)
            mcp = np.random.uniform(param.mcp_scope[0], param.mcp_scope[1], 6)
            prey = np.random.choice(Herbiovores)
            predators[i] = Predator(position, adaptation, mcp , 0, prey)
            
        return predators
    
    def create_herbivore_population(self, Plants):
        herbivores = np.empty(shape=(param.n_he,), dtype=object)
        
        for i in range(param.n_he):
            position = self.deploy_position()
            adaptation = self.count_adaptation(position)
            mcp = np.random.uniform(param.mcp_scope[0], param.mcp_scope[1], 6)
            prey = np.random.choice(Plants)
            herbivores[i] = Herbivore(position, adaptation, mcp, 0, prey)
            
        return herbivores
    
    def create_plant_population(self):  
        plants = np.empty(shape=(param.n_pl,), dtype=object)
        
        for i in range(param.n_pl):
            position = self.deploy_position()
            adaptation = self.count_adaptation(position)
            plants[i] = Plant(position, adaptation)
        
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
    def __init__(self, position, adaptation):
        self.position = position
        self.adaptation = adaptation
        self._best_organism = None

    @property
    def best_organism(self):
        return self._best_organism     
    
    def save_best_organism(self):
        if self.best_organism is not None:
            if self.best_organism[0] > self.adaptation:
                self._best_organism = [self.adaptation, self.position]
        else:
            self._best_organism = [self.adaptation, self.position]
    
    def set_adaptation(self, adaptation):
        self.adaptation = adaptation
        
    def set_position(self, position):
        self.position = position

class Plant(Organism):
    def __init__(self, position, adaptation):
        super().__init__(position, adaptation)
        self._size = None
        
    @property
    def size(self):
        return self._size
    
    def set_size(self, size):
        self._size = size
        
class Animal(Organism):
    def __init__(self,position, adaptation, mcp, velocity, prey):
        super().__init__(position, adaptation)
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
        
    def set_prey(self, prey):
        self.prey = prey
        
class Herbivore(Animal):
    def __init__(self,position, adaptation, mcp, velocity, prey):
        super().__init__(position, adaptation, mcp, velocity, prey)       
    

class Predator(Animal):
    def __init__(self,position, adaptation, mcp, velocity, prey):
        super().__init__(position, adaptation, mcp, velocity, prey)
        self.life = param.life
        
    def set_life(self, life):
        self.life = life
        
def main():
    data = d.Dataset()
    input, output = data.get_dataset()
    neural_network = nn.NeuralNetwork(input, output)
    eco = EcosystemAlgorithm(neural_network)

    plants, herbivores, predators = eco.create_population()

    eco.save_bests(plants, herbivores, predators)
    
    for i in tqdm(range(param.alg_iter)):
        eco.perform_movement(herbivores, eco.best_herbivore, eco.best_plant)
        eco.perform_movement(predators, eco.best_predator, eco.best_herbivore)
        
        if i % param.interactions == 0:
            eco.exploration_exploitation_change()
            plants = eco.plant_herbivore_simulation(plants, herbivores)
            herbivores, predators = eco.herbivore_predator_simulation(herbivores, predators)

            plants = eco.plant_crossover(plants)
            herbivores = eco.animal_crossover(Herbivore, herbivores, plants, param.n_he)
            predators = eco.animal_crossover(Predator, predators, herbivores, param.n_pr)
            
            plants = eco.plant_mutation(plants)
            herbivores = eco.animal_mutation(herbivores, param.mutation_he)
            predators = eco.animal_mutation(predators, param.mutation_pr)
    
            eco.save_bests(plants, herbivores, predators)
            
        if i % param.save_to_file == 0 or i == param.alg_iter - 1:
            first_use = True if i == 0 else False 
            eco.print_loss(plants, herbivores, predators, i, first_use)
        
        eco.set_adaptation_for_all(plants, herbivores, predators)
        eco.save_bests(plants, herbivores, predators)
        
    print('deaths: {}\tmutations: {}'.format(eco.deaths, eco.mutations))

if __name__ == "__main__":
    main()