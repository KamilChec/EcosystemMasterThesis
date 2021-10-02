import numpy as np
import sklearn
import sklearn.datasets
import matplotlib.pyplot as plt

class Dataset:
    def __init__(self):
        np.random.seed(0)
        self.input, self.output = sklearn.datasets.make_moons(200, noise=0.20)
        
    def get_dataset(self):
        return self.input, self.output
    
    def plot_decision_boundary(self, pred_func):
        X, y = self.input, self.output
        x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
        y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
        h = 0.01
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
        plt.plot()
        