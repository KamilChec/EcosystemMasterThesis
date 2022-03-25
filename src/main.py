import dataset as d
import neuralNetwork as nn
import geneticAlgorithms as ga
import PSO as pso
from tqdm import tqdm

def main():
    data = d.Dataset()
    input, output = data.get_dataset()
    neural_network = nn.NeuralNetwork(input, output)
    genetic_algorithm = ga.GeneticAlgorithm(100, neural_network)
    partical_optimalization =  pso.ParticleSwarmOptimization(10000, 0.8, 1, 0, neural_network)
    
    
    # chromosomes = genetic_algorithm.create_population()
    # genetic_algorithm.set_adaptation(chromosomes)
    # genetic_algorithm.print_loss(chromosomes, 'init', first_use=True)
    # for i in tqdm(range(20000)):
    #     crossedOffspring = genetic_algorithm.crossover(chromosomes)
    #     mutatedOffspring = genetic_algorithm.mutation(crossedOffspring)
    #     genetic_algorithm.set_new_population(mutatedOffspring, chromosomes)
    #     if i % 1000 == 0:
    #             genetic_algorithm.print_loss(chromosomes, i)
    # print(genetic_algorithm.best_chromosome(chromosomes))
    # genetic_algorithm.plot_evolution_of_adaptation()
    # data.plot_decision_boundary(lambda x: neural_network.predict(x))
    
    # population = partical_optimalization.create_population()
    # partical_optimalization.set_all_parameters(population)
    # for i in tqdm(range(100)):
    #     partical_optimalization.set_all_parameters(population)
    #     partical_optimalization.perform_movement(population)
    #     if i % 10 == 0:
    #         if (i == 0): 
    #             partical_optimalization.print_loss(i, first_use=True)
    #         else:
    #             partical_optimalization.print_loss(i)
    #         if partical_optimalization.c1 > 0.1: partical_optimalization.c1 -= 0.1
    #         partical_optimalization.c2 += 0.1
            
    # partical_optimalization.plot_evolution_of_adaptation()
    # data.plot_decision_boundary(lambda x: neural_network.predict(x))
    
    
    neural_network = nn.NeuralNetwork(input, output)
    neural_network.build_model_with_backpropagation( print_loss=True)
    print("Wrong predictions: {}".format(neural_network.count_wrong_predictions()))
    # print(neural_network)
    neural_network.plot_evolution_of_adaptation()
    data.plot_decision_boundary(lambda x: neural_network.predict(x))



if __name__ == "__main__":
    main()