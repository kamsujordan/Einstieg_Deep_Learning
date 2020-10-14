import numpy as np

def init_variables():
    """
        Init model Variables (weights, bias)
    """
    weights = np.random.normal(size=2)
    bias = 0
    
    return weights, bias

def get_dataset():
    """
        METHOD USED TO GENERATE THE DATASET
    """

    #Numbers of row per class
    row_per_class = 5
    #Generate rows
    sick = np.random.randn(row_per_class, 2) + np.array([-2, -2])
    healthy = np.random.randn(row_per_class, 2) + np.array([2, 2])

    features = np.vstack([sick, healthy])
    targets = np.concatenate((np.zeros(row_per_class), np.zeros(row_per_class) + 1))

    return features, targets
    """
        Print the Shape of the matrix
        print (np.random.randn(row_per_class, 2).shape)
    """

def pre_activation(features, weights, bias):
    """
        COMPUTE PRE ACTIVATION
    """
    return np.dot(features, weights) + bias

def activation(z):
    """
        compute activation
    """
    return 1 / (1+ np.exp(-z))

if __name__== '__main__':
    #Dataset
    features, targets = get_dataset()
    # Variables
    weights, bias = init_variables()
    #compute pre activation
    z = pre_activation(features, weights, bias)
    #conpute activation
    a = activation(z)

    print(z)
    print(targets)
    print(a)