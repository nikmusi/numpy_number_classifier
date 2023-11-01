import numpy as np
from loss_functions.utils import softmax, one_hot

class CrossEntropyLoss:
    def __init__(self):
        self.A_softmax = None
        self.loss = None
    
    def calc_loss(self, Y_pred, Y_gt):
        """Calculates the loss of the given forward pass result, by calculating the softmax and applying the 
        function of the Cross Entropy Loss.

        Args:
            Y_pred (np.array): the output of the last layer inside the neural net
            Y_gt (np.array): the actual labels of the given input

        Returns:
            float: the Cross Entropy Loss of the given predictions
        """
        self.A_softmax = softmax(Y_pred)
        
        m = Y_gt.size
        # cap maximum loss to loss = 9 ( log(1e-9)=9 )
        p = np.maximum(self.A_softmax[Y_gt, range(m)], 1e-9)
        
        log_likelihood = - np.log(p)
        loss = 1/m * np.sum(log_likelihood)
        
        return loss
    
    def backward(self, Y):
        """Applies the backprob of the Cross Entropy Loss

        Args:
            Y (np.array): the actual labels of the input images

        Returns:
            np.array: the derivative of the Loss in respect to the output of the last network layer
        """
        one_hot_Y = one_hot(Y)
        return self.A_softmax - one_hot_Y
