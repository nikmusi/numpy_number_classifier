import numpy as np
import multiprocessing as mp

class Conv2d:    
    def __init__(self, img_size, kernel_depth, kernel_size):
        
        # init arguments !ONLY QUADRATIC KERNELS - only 1d input!
        self.img_size = img_size
        self.kernel_depth = kernel_depth
        self.kernel_size = kernel_size
        self.output_size = img_size - (kernel_size - 1)
        
        # init parameters
        self.weights = np.random.rand(kernel_depth, kernel_size, kernel_size) - 0.5
        self.bias = np.random.rand(kernel_depth, self.output_size, self.output_size) - 0.5
        self.delta_weights = None
        self.delta_bias = None
        
        self.X = None
        self.Y = None      
    
    def forward(self, X):
        """Applies the forward pass with the current weights and bias of the layer.

        Args:
            X (np.array): input of the conv layer in shape:(batch_size, kernel_depth, input_width, 
            input_height)

        Returns:
            np.array: the output of the layer
        """    
        self.X = X  
        self.X.reshape(X.shape[0], X.shape[1], X.shape[2]) 
             
        # generate the output array, already holding the bias
        self.Y = np.zeros((X.shape[0], self.kernel_depth, self.output_size, self.output_size))
        self.Y[np.arange(X.shape[0]), :, :, :] = self.bias
        
        # calculate how many steps the kernel will make (stride always = 1)
        conv_steps = self.output_size**2

        # do the farward step
        for i in range(self.kernel_depth):
            
            # my solution for conv - not that great --> use scipy
            for j in range(conv_steps):
                    # calc current position of the kernel
                    row_k = int(j / self.output_size)
                    col_k = j % self.output_size
                    
                    # calc the correlation of the position over all images in X
                    y = self.X[:,row_k:row_k+self.kernel_size,col_k:col_k+self.kernel_size,0] * \
                        self.weights[i,:,:]
                    y = y.reshape(self.X.shape[0], self.kernel_size**2).sum(axis=1)
                    
                    self.Y[:, i,  row_k, col_k] += y
                
        return self.Y
    
    def backward(self, dY):
        """Aplies the backprob of the convolutional layer, storing the Loss derivatives in respect to weights
        and biases as lokal parameters.

        Args:
            dY (np.array): the derivative of the Loss in respect to the output of this Conv2d layer

        Returns:
            np.array: the derivative of the Loss in respect to the input of this Conv2d layer
            
        TODO: Add the calculation dL/dX --> needed if a conv layer isn't the first layer
        """
        conv_steps = self.kernel_size**2    # kernel size
        full_conv_steps = self.img_size**2  # size of input
        
        # init/reset expected output --> like opt.zero_grad()
        self.delta_weights = np.zeros((self.kernel_depth, self.kernel_size, self.kernel_size))
        
        # do the backwards step
        for i in range(self.kernel_depth):
            
            # calculate the delta_weights
            for j in range(conv_steps):
                # calc current position of the kernel
                row_k = int(j / self.kernel_size)
                col_k = j % self.kernel_size
                
                # calc the correlation of the position over all images in X
                y = self.X[:,row_k:row_k+self.output_size,col_k:col_k+self.output_size,0] *\
                    dY[:,i,:,:]
                y = y.reshape(self.X.shape[0], self.output_size*self.output_size).sum(axis=1)
                y = y.sum() / self.X.shape[0]
                
                self.delta_weights[i,  row_k, col_k] = y
        
        self.delta_bias = dY.sum(axis=0) / self.X.shape[0]
    
    def update(self, learning_rate):
        """Updates the weights and biases based on the last backward pass

        Args:
            learning_rate (float): learing rate that will be applied
        """
        self.weights -= learning_rate * self.delta_weights
        self.bias -= learning_rate * self.delta_bias
    
    def load_weights(self, weights, bias):
        """Loads given weights and bias into the layer

        Args:
            weights (np.array): the weights that will be stored as the current parameters
            bias (np.array)): the bias that will be stored as the current parameters
        """
        self.weights = weights
        self.bias = bias
        