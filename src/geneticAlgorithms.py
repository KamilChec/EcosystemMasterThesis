class GeneticAlgorithm:
    def __init__(self):
        pass
    
    def create_population(self):
        pass
    
    def adaptation(self):
        pass
    
    def probability(self):
        pass
    
    def choose_parents(self):
        pass
    
    def selection(self):
        pass
    
    def crossover(self):
        pass
    
    def mutation(self):
        pass
    
    def best_chromosome(self):
        pass
    
class Chromosome:
    def __init__(self, id, body, adaptation):
        self.id = id
        self.body = body
        self.adaptation = adaptation
    
    def set_adaptation(self, adaptation):
        self.adaptation = adaptation
    
    def set_body(self, body):
        self.body = body