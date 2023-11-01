import numpy as np

def softmax(X):
        """Calculates the softmax of the given network-output

        Args:
            X (np.array): output of the previouse layer

        Returns:
            np.array: activation of the previous layer
        """
        X = np.float64(X)
        e = np.exp(X - np.max(X, axis=0))
        return e / np.sum(e, axis=0)
    
def one_hot(Y):
        """Gererates a matrix, which stores a vector corresponding to each label given in Y. Each vector holds
        zeros everywhere, except the dimension of the corresponding label inside Y.

        Args:
            Y (np.array): the actual labels of the input images

        Returns:
            np.array: the actual labels of the input images as one_hot_Y
        """
        one_hot = np.zeros((Y.size, 10))
        one_hot[np.arange(Y.size), Y] = 1
        one_hot = one_hot.T
        return one_hot