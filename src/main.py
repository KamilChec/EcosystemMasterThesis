import dataset as d
import neuralNetwork as nn
import geneticAlgorithms as ga

def main():
    data = d.Dataset()
    input, output = data.get_dataset()
    neural_network = nn.NeuralNetwork(input, output)
    genetic_algorithm = ga.GeneticAlgorithm(100, neural_network)
    
    chromosomes = genetic_algorithm.create_population()
    genetic_algorithm.set_adaptation(chromosomes)
    for _ in range(1000):
        offspring = genetic_algorithm.crossover(chromosomes)
        offspring = genetic_algorithm.mutation(offspring)
        genetic_algorithm.set_new_population(offspring, chromosomes)
    print(genetic_algorithm.best_chromosome(chromosomes))
    
    # model.build_model_with_backpropagation(3, print_loss=True)
    # data.plot_decision_boundary(lambda x: model.predict(x))



if __name__ == "__main__":
    main()