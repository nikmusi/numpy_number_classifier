import numpy as np

class Linear():
    def __init__(self, input_dim, output_dim):
        
        self.X_dim = input_dim
        self.Y_dim = output_dim
        
        # init weigths and biases -> deltas will be holding Loss in deriv to weights/biases
        self.weights = np.random.rand(output_dim, input_dim) - 0.5
        self.bias = np.random.rand(output_dim, 1) - 0.5
        self.delta_weights = None 
        self.delta_bias = None

        self.X = None
        self.Y = None
    
    def forward(self, X):
        """Applies the forward pass with the current weights and bias of the layer.

        Args:
            X (np.array): input of the Linear layer in shape:(img_size, batch_size)

        Returns:
            np.array: the output of the layer
        """
        self.X = X
        self.Y = self.weights.dot(self.X) + self.bias
        return self.Y
    
    def backward(self, dY):
        """Aplies the backprob of the Linear layer, storing the Loss derivatives in respect to weights
        and biases as lokal parameters.

        Args:
            dY (np.array): the derivative of the Loss in respect to the output of this Linear layer

        Returns:
            np.array: the derivative of the Loss in respect to the input of this Linear layer
        """
        m = self.X.shape[0]     # number of images
        
        # calculate derivatives in respect to weights, bias and input
        self.delta_weights = 1/m * dY.dot(self.X.T)
        self.delta_bias = 1/m * np.sum(dY, 1)
        delta_X = self.weights.T.dot(dY)
        return delta_X 
    
    def update(self, learning_rate):
        """Updates the weights and biases based on the last backward pass

        Args:
            learning_rate (float): learing rate that will be applied
        """
        self.weights -= learning_rate * self.delta_weights
        self.bias -= learning_rate * self.delta_bias.reshape(self.bias.shape)
        
    def load_weights(self, weights, bias):
        """Loads given weights and bias into the layer

        Args:
            weights (np.array): the weights that will be stored as the current parameters
            bias (np.array)): the bias that will be stored as the current parameters
        """
        self.weights = weights
        self.bias = bias
        
    