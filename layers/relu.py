import numpy as np

class ReLU():
    def __init__(self):
        
        self.X = None
        self.Y = None
        pass
    
    def forward(self, X):
        """Applies the activation function of the Rectified Linear Unit

        Args:
            X (np.array): the input of the ReLU

        Returns:
            np.array: activation of the given input
        """
        self.X = X
        self.Y = np.maximum(0, X)
        return self.Y
    
    def backward(self, dY):
        """Aplies the backprob of the Rectified Linear Unit

        Args:
            dY (np.array): the derivative of the Loss in respect to the output of this ReLU layer

        Returns:
            np.array: the derivative of the Loss in respect to the input of this ReLU layer
        """
        return dY * (self.X > 0)