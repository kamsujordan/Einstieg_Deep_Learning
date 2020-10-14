if __name__ == '__main__':
    #Function to minimize
    fc = lambda x,y: (3*x**2)  + (5*y**2) + (x*y)
    #Set partial derivatives
    partial_derivative_x = lambda x, y: (6*x) + y
    partial_derivative_y = lambda x, y: (10*y) + x
    #set variables
    x= 10
    y= -13
    #Learning rate
    learning_rate = 0.1
    print ("Fc = %s" % (fc(x,y)))

    #one epoch is  one period of minimisation
    for epoch in range(0, 20):
        #COMPUTE GRADIENTS
        x_gradient = partial_derivative_x(x, y)
        y_gradient = partial_derivative_y(x,y)
        #APPLY GRADIENT DESCENT
        x = x - learning_rate * x_gradient
        y = y - learning_rate * y_gradient
        #KEEP TRACK OF THE FUNCTION VALUE
        print ("Fc = %s" % (fc(x,y)))

    #PRINT FINAL VARIABLE VALUES
    print ("")
    print ("x = %s" %x)
    print ("y = %s" % y)