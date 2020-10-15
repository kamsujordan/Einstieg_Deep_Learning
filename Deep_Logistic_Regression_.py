import numpy as np
import matplotlib.pyplot as plt



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
    row_per_class = 100
    #Generate rows
    sick = np.random.randn(row_per_class, 2) + np.array([-2, -2])
    sick2 = np.random.randn(row_per_class, 2) + np.array([2, 2])

    healthy = np.random.randn(row_per_class, 2) + np.array([-2, 2])
    healthy2 = np.random.randn(row_per_class, 2) + np.array([2, -2])

    features = np.vstack([sick, sick2, healthy, healthy2])
    targets = np.concatenate((np.zeros(row_per_class * 2), np.zeros(row_per_class * 2) + 1))

    return features, targets
    """
        Print the Shape of the matrix
        print (np.random.randn(row_per_class, 2).shape)
    """



def pre_activation(features, weights, bias):
    """
        COMPUTE THE PRE ACTIVATION
        **input: **
            *features: (Numpy Matrix)
            *weights: (Numpy  vector)
            *bias: (Integer)
    """
    return np.dot(features, weights) + bias



def activation(z):
    """
        Activation/Sigmoid Function
        **inout: **
        *z: (Integer|Numpy Array)
    """
    return 1 / (1+ np.exp(-z))



def derivative_activation(z):
    """
        compute the derivative of the activation (derivative of a Sigmoid Function)
    """
    return activation(z) * (1 - activation(z))


def train(features, targets, weights, bias):
    """
        Method used to train the model using the gradient descent method
        
        **input: **
            *features: (Numpy Matrix)
            *targets: (Numpy vector)
            *weights: (Numpy vector)
            *bias: (Integer)
        **return (Numpy vector, Numpy vector) **
            *update weights
            *update bias
    """

    epochs = 100
    learning_rate = 0.1

    #print current Accuracy
    #Accuracy shows how good the Algorithm worked
    predictions = predict(features, weights, bias)
    print( "Accuracy", np.mean(predictions == targets))

    #plot points
    plt.scatter(features[:, 0], features[:, 1], s=40, c= targets, cmap = plt.cm.Spectral)
    plt.show()

    for epoch in range(epochs):
        if epoch % 10 == 0:
            predictions = activation(pre_activation(features, weights, bias))
            print("Cost = %s" % cost(predictions, targets))
            
        #Init gradients
        weights_gradients = np.zeros(weights.shape)
        bias_gradient = 0
        #Go through each row
        for feature, target in zip(features, targets):
            #compute prediction
            z = pre_activation(feature, weights, bias)
            y = activation(z)
            #Update gradients
            weights_gradients += (y - target) * derivative_activation(z) * feature
            bias_gradient += (y - target) * derivative_activation(z)
         #Update variables
        weights = weights - learning_rate * weights_gradients
        bias = bias - learning_rate * bias_gradient

    #print current Accuracy
    predictions = predict(features, weights, bias)
    print( "Accuracy", np.mean(predictions == targets))



def predict(features, weights, bias):
    """
        
    """

    z = pre_activation(features, weights, bias)
    y= activation(z)
    return (np.round(y))



def cost(predictions, targets):
    """
        Compute the cost of the model
        **input: **
            *predictions: (Numpy vector) y
            *targets: (Numpy vector) t
    """

    return np.mean((predictions - targets)**2)



if __name__== '__main__':
    #Dataset
    features, targets = get_dataset()
    # Variables
    weights, bias = init_variables()
    #compute pre activation
    z = pre_activation(features, weights, bias)
    #conpute activation
    a = activation(z)

    train(features, targets, weights, bias)
