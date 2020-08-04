from matplotlib import pyplot as plt
from matplotlib import style
import numpy as np


def print_graph(X, Y):

    #https://matplotlib.org/tutorials/introductory/pyplot.html
    style.use ('classic')
    plt.scatter(Y, X, color='r')
    plt.plot()

def print_line(slope, intercept):
    axes = plt.gca()
    plt.xlabel('FEATURE')
    plt.ylabel('TARGET')
    x_vals = np.array(axes.get_xlim())
    y_vals = slope + intercept * x_vals
    plt.plot(x_vals, y_vals, color = 'b')


def print_graph_line(Y, X, slope, intercept):
    axes = plt.gca()
    plt.xlabel('FEATURE')
    plt.ylabel('TARGET')

    style.use ('classic')
    plt.scatter(Y, X, color='r')

    x_vals = np.array(axes.get_xlim())
    y_vals = slope + intercept * x_vals
    plt.plot(x_vals, y_vals, color = 'b')

def show():
    plt.show()

def Polynomial(thetas, min, max):

    X = np.linspace(min, max, 10, endpoint=True)
    Y = list()
    
    plt.xlabel('FEATURE')
    plt.ylabel('TARGET')
    style.use ('classic')

    for x in X:
        value = 0
        for index, coeff in enumerate(thetas):
            value += coeff * x ** index
        Y.append(value)
    plt.plot(X, Y, color = 'b')
