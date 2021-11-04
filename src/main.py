import dataset as d
import neuralNetwork as nn
import geneticAlgorithms as ga
from tqdm import tqdm

def main():
    data = d.Dataset()
    input, output = data.get_dataset()
    neural_network = nn.NeuralNetwork(input, output)
    genetic_algorithm = ga.GeneticAlgorithm(50, neural_network)
    
    chromosomes = genetic_algorithm.create_population()
    genetic_algorithm.set_adaptation(chromosomes)
    genetic_algorithm.print_loss(chromosomes, 'init', first_use=True)
    for i in tqdm(range(2000)):
        crossedOffspring = genetic_algorithm.crossover(chromosomes)
        mutatedOffspring = genetic_algorithm.mutation(crossedOffspring)
        genetic_algorithm.set_new_population(mutatedOffspring, chromosomes)
        if i % 100 == 0:
                genetic_algorithm.print_loss(chromosomes, i)
    print(genetic_algorithm.best_chromosome(chromosomes))
    genetic_algorithm.plot_evolution_of_adaptation()
    data.plot_decision_boundary(lambda x: neural_network.predict(x))
    
    # neural_network = nn.NeuralNetwork(input, output)
    # neural_network.build_model_with_backpropagation(3, print_loss=True)
    # data.plot_decision_boundary(lambda x: neural_network.predict(x))



if __name__ == "__main__":
    main()