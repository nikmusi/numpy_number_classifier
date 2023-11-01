import numpy as np
from tqdm.auto import tqdm
import pickle
import matplotlib.pyplot as plt

from data_loader import DataLoader
from layers.conv2d import Conv2d
from layers.linear import Linear
from layers.relu import ReLU
from loss_functions.cross_entropy_loss import CrossEntropyLoss


def flatten(A):
    """Flattens the output of a convolutional layer, that a linear layer can process it.

    Args:
        A (np.array): output of a conv2d layer

    Returns:
        np.array: flattend array
    """
    flat_A = A.reshape(A.shape[0], A.shape[1]*A.shape[2]*A.shape[3])
    return flat_A.T     

def get_predictions(A):
    """Returns the classification with the highest score

    Args:
        A (np.array): output of the neural net, holding scores for each label 
        
    Returns:
        np.array: holding the predicted label (highest score) for every image
    """
    return(np.argmax(A, 0))

def get_accuracy(predictions, Y):
    """Calculates the accuracy of the given predictions

    Args:
        predictions (np.array): predicted classifications
        Y (np.array): ground truth labels

    Returns:
        float: the accuracy of the predictions [0,1]
    """
    return np.sum(predictions == Y) / Y.size

def forward(X):
    """The forward pass of the neural net defined in main. This function needs to be updated if the layer 
    structure of the net changes.

    Args:
        X (np.array): input images loaded from the dataloader

    Returns:
        np.array: raw output of the forward pass
    """
    
    # forward convolutional layer
    Z1 = c1.forward(X)
    A1 = r1.forward(Z1)
    # flatten the layer
    A1_flat = flatten(A1)
    # forward linear layers
    Z2 = l1.forward(A1_flat)
    A2 = r2.forward(Z2)
    Z3 = l2.forward(A2)
    
    return Z3

def backpropagation(Y, dZ3):
    """Applies the backpropagation for every layer of the net

    Args:
        Y (np.array): labels loaded by the dataloader
        dZ3 (np.array): output generated by the forward pass
    """
    # backprob of linear layers
    dA2 = l2.backward(dZ3)
    dZ2 = r2.backward(dA2)
    dA1_flat = l1.backward(dZ2)
    # reshape the flat delta_activation to the conv-shape (inverse the flatten())
    dA1_flat = dA1_flat.T
    dA1 = dA1_flat.reshape(dA1_flat.shape[0], c1.kernel_depth, c1.output_size, c1.output_size)
    # backprob of conv layer
    c1.backward(dA1)
    
def update_params(lr):
    """Updates the weights and biases of every layer in the net. This function has to be updated if the
    layer structure of the net changes.

    Args:
        lr (float): the learning rate, that will be applied
    """
    c1.update(lr)
    l1.update(lr)
    l2.update(lr)
    
def store_weights(epoch):
    """Stores the current weights of every layer inside a pickle file. This function has to be updated, when
    the layer structure of the net changes. The weights will be stored in data/weights. Old weights will be 
    overwritten.
    
    Args:
        epoch (int): current epoch
    """
    weights = {
        "c1": {
            "weights": c1.weights,
            "bias": c1.bias
        },
        "l1": {
            "weights": l1.weights,
            "bias": l1.bias
        },
        "l2": {
            "weights": l2.weights,
            "bias": l2.bias
        }
    }
    
    with open(f'data/weights/epoch_{epoch}.pickle', 'wb') as f:
        pickle.dump(weights, f, protocol=pickle.HIGHEST_PROTOCOL)

def train(batch_size, epochs, learning_rate):
    """Trains the neural network with the given hyperparameters.

    Args:
        batch_size (int): size of batches
        epochs (int): number of epochs
        learning_rate (float): learning rate
    """
    # specify some hyperparameters and choose loss fn
    loss_fn = CrossEntropyLoss()
    
    # load the data
    dataloader = DataLoader(batch_size)
    dataloader.load_mnist_images('data/dataset/train-images-idx3-ubyte.gz')
    dataloader.load_mnist_labels('data/dataset/train-labels-idx1-ubyte.gz')
    
    # start the training process
    for i in range(epochs):
        
        loss = 0
        print(f"Calculating epoch {i}/{epochs}")
        for batch in tqdm(range(0,60000, batch_size)):
            
            X_train, Y_train = dataloader.get_batch(batch)
            Z3 = forward(X_train)
            loss = loss_fn.calc_loss(Z3, Y_train)
            
            # do the backward pass
            dZ3 = loss_fn.backward(Y_train) 
            backpropagation(Y_train, dZ3)
            update_params(learning_rate)
            
            loss += loss
            
        # reset dataloader for the next pass and print epoch results
        dataloader.reset() 
        print(f"Average loss in epoch {i} was: {loss / len(range(0,60000, batch_size))}")
        store_weights(i)
        
def test():
    """Runs the test data trough the trained neural net

    Returns:
        list: [images, predictions, labels] containing the image data, predicted label and gt-label of 
        every image in the test dataset
    """
    # load the data
    dataloader = DataLoader(batch_size=32)
    dataloader.load_mnist_images('data/dataset/train-images-idx3-ubyte.gz', train=False)
    dataloader.load_mnist_labels('data/dataset/train-labels-idx1-ubyte.gz', train=False)
    loss_fn = CrossEntropyLoss()
    
    # Load weights
    with open(f'data/weights/epoch_9.pickle', 'rb') as f:
        weights = pickle.load(f)
    c1.load_weights(weights["c1"]["weights"], weights["c1"]["bias"])
    l1.load_weights(weights["l1"]["weights"], weights["l1"]["bias"])
    l2.load_weights(weights["l2"]["weights"], weights["l2"]["bias"])
    
    accuracy = 0
    images = []
    predictions = []
    labels = []
    
    # batches are not needed in the test run (no backward passing)...
    # but the data loader returns only batches for now...
    for batch in tqdm(range(0,10000, 32)):
        
        # load the current batch
        X_test, Y_test = dataloader.get_batch(batch)
        # do the forward pass and calculate the loss to apply softmax
        Z3 = forward(X_test)
        loss = loss_fn.calc_loss(Z3, Y_test)
        # calculate and store accuracy
        preds = get_predictions(loss_fn.A_softmax)
        accuracy += get_accuracy(preds, Y_test)
        
        # append the images, preds, and labels for plotting specific results later on if needed
        for i in range(32):
            images.append(X_test[i])
            predictions.append(preds[i])
            labels.append(Y_test[i])
    
    accuracy = accuracy / len(range(0,10000, 32))
    print(f"The accuracy on the test-set was: {accuracy}")
    
    return images, predictions, labels
    
    
def test_specific_prediction(index, images, predictions, labels):
    """Prints the test results of a specific image and displays the test img on the screen.

    Args:
        index (int): index of an image inside the test dataset
        images (list): images of the test-data 
        predictions (list): predicted labels of the test data
        labels (list): actual labels of the test data
    """
    print(f"Prediction: {predictions[index]}")
    print(f"Label: {labels[index]}")
    # plot the image
    image = np.asarray(images[index]).squeeze()
    plt.imshow(image, cmap="gray")
    plt.show()
    
    
if __name__ == "__main__":
    
    # init the layers of the neural net
    c1 = Conv2d(28, 8, 3)
    r1 = ReLU()
    l1 = Linear(8*26*26, 500)
    r2 = ReLU()
    l2 = Linear(500, 10)
    
    train(batch_size=32, epochs=10, learning_rate=1e-4)
    
    # imgs, preds, labels = test()
    # test_specific_prediction(20, imgs, preds, labels)
                  