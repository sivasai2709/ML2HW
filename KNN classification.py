from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import numpy as np

# Load dataset
iris = load_iris()
X = iris.data[:, :2]   # Only sepal length & sepal width
y = iris.target

# Plot function
def plot_knn(k):
    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(X, y)

    # Mesh grid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05),
                         np.arange(y_min, y_max, 0.05))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k')
    plt.title(f"k={k}")
    plt.xlabel("Sepal length")
    plt.ylabel("Sepal width")
    plt.show()

for k in [1,3,5,10]:
    plot_knn(k)
