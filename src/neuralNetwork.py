import numpy as np
import matplotlib.pyplot as plt

np.seterr(all='ignore')

class NeuralNetwork:
    nn_input_dim = 2 # input layer dimensionality
    nn_output_dim = 2 # output layer dimensionality
    nn_hdim = 3

    # Gradient descent parameters (I picked these by hand)
    epsilon = 0.01 # learning rate for gradient descent
    reg_lambda = 0.01 # regularization strengt
    
    all_generation_loss = []
    
    def __init__(self, input, output):
        self.input = input
        self.output = output
        self.num_examples = len(self.input)
        
        self.W1 = 0
        self.b1 = 0
        self.W2 = 0
        self.b2 = 0
    
    def forward_propagation(self, x=None):
        if x is None:
            x = self.input
        W1, b1, W2, b2 = self.W1, self.b1, self.W2, self.b2
        
        z1 = x.dot(W1) + b1
        a1 = np.tanh(z1)
        z2 = a1.dot(W2) + b2
        exp_scores = np.exp(z2)
        prediction = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)  
        return prediction        
    
    def calculate_loss(self):
        y = self.output
        W1, W2 = self.W1, self.W2
        num_examples = self.num_examples
        
        probs = self.forward_propagation()
        corect_logprobs = -np.log(probs[range(num_examples), y])
        data_loss = np.sum(corect_logprobs)
        data_loss += self.reg_lambda/2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
        return 1./num_examples * data_loss
    
    def predict(self, x):
        probs = self.forward_propagation(x)
        return np.argmax(probs, axis=1)
    
    def set_network_parameters(self, W1, b1, W2, b2):
        self.W1 = W1
        self.b1 = b1
        self.W2 = W2
        self.b2 = b2
        
    def backpropagation(self, probs):
        X, y = self.input, self.output
        W1, b1, W2, b2 = self.W1, self.b1, self.W2, self.b2
        epsilon = self.epsilon
        
        delta3 = probs
        delta3[range(self.num_examples), y] -= 1
        z1 = X.dot(W1) + b1
        a1 = np.tanh(z1)
        dW2 = (a1.T).dot(delta3)
        db2 = np.sum(delta3, axis=0, keepdims=True)
        delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2))
        dW1 = np.dot(X.T, delta2)
        db1 = np.sum(delta2, axis=0)

        dW2 += self.reg_lambda * W2
        dW1 += self.reg_lambda * W1
        
        W1 += -epsilon * dW1
        b1 += -epsilon * db1
        W2 += -epsilon * dW2
        b2 += -epsilon * db2
        return W1, b1, W2, b2
    
    def print_loss(self, index):
        loss = self.calculate_loss()
        self.all_generation_loss.append(loss)
        print("Loss after iteration %i: %f" %(index, loss))
        
    def create_population(self, nn_hdim):
        # np.random.seed(0)
        W1 = np.random.randn(self.nn_input_dim, nn_hdim) / np.sqrt(self.nn_input_dim)
        # b1 = np.zeros((1, nn_hdim)) zmiana dla PSO
        b1 = np.random.randn(1, nn_hdim)
        W2 = np.random.randn(nn_hdim, self.nn_output_dim) / np.sqrt(nn_hdim)
        # b2 = np.zeros((1, self.nn_output_dim)) zmiana dla PSO
        b2 = np.random.randn(1, self.nn_output_dim)

        return W1, b1, W2, b2
        
        
    def build_model_with_backpropagation(self, nn_hdim, num_passes=20000, print_loss=False):
        W1, b1, W2, b2 = self.create_population(nn_hdim)
        self.set_network_parameters(W1, b1, W2, b2)
        
        for i in range(0, num_passes):
            probs = self.forward_propagation()
            W1, b1, W2, b2 = self.backpropagation(probs)
            self.set_network_parameters(W1, b1, W2, b2)
            
            if print_loss and i % 1000 == 0:
                self.print_loss(i)
    
    def plot_evolution_of_adaptation(self):
        nn_losses = len(self.all_generation_loss)
        x = np.linspace(0, nn_losses, nn_losses)
        plt.plot(x, self.all_generation_loss, '-o')
        plt.show()
        
    def __str__(self):
      return 'Neural networl: W1: \n' + str(self.W1) + '\nb1: \n' + str(self.b1) + '\nW2: \n' + str(self.W2) + '\nb2: \n' + str(self.b2) + '\n' 