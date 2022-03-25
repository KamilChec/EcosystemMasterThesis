import numpy as np
import sklearn
import sklearn.datasets
import matplotlib.pyplot as plt

class Dataset:
    def __init__(self):
        np.random.seed(0)
        # self.input, self.output = sklearn.datasets.make_moons(200, noise=0.1)
        self.input, self.output = sklearn.datasets.make_circles(100, noise=0.06)
        # self.input, self.output = sklearn.datasets.make_blobs(200, centers=4, center_box=(-15,15))
        # self.input, self.output = sklearn.datasets.make_blobs(200, centers=4)
        # self.input, self.output = sklearn.datasets.make_classification(
        #     n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=1
        # )
        # rng = np.random.RandomState(2)
        # self.input += 2 * rng.uniform(size=self.input.shape)
        
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
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral, edgecolors="k")
        plt.title('Decision boundary')
        plt.show()
