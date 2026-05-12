# utilities.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.inspection import DecisionBoundaryDisplay

def visualize_classifier(classifier, X, y, title=''):
    min_x, max_x = X[:, 0].min() - 1.0, X[:, 0].max() + 1.0
    min_y, max_y = X[:, 1].min() - 1.0, X[:, 1].max() + 1.0

    mesh_step_size = 0.01
    x_vals, y_vals = np.meshgrid(
        np.arange(min_x, max_x, mesh_step_size),
        np.arange(min_y, max_y, mesh_step_size)
    )

    output = classifier.predict(np.c_[x_vals.ravel(), y_vals.ravel()])
    output = output.reshape(x_vals.shape)

    plt.figure()
    plt.title(title)

    plt.pcolormesh(x_vals, y_vals, output, cmap=plt.cm.gray)

    markers = 's', 'o', '^', 'v', 'D'
    colors  = 'black', 'white', 'gray', 'lightgray', 'darkgray'

    for i, (m, c) in enumerate(zip(markers, colors)):
       mask = (np.array(y) == i)
    if mask.any():
            plt.scatter(X[mask, 0], X[mask, 1],
                        s=75, facecolors=c, edgecolors='black',
                        linewidth=1, marker=m)

    plt.xlim(min_x, max_x)
    plt.ylim(min_y, max_y)
    plt.xticks(())
    plt.yticks(())