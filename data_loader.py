import gzip
import numpy as np
import pickle


class DataLoaderMNIST:
    def __init__(self, image_root, label_root, train=True):
        self.image_root = image_root
        self.label_root = label_root
        self.train = train

    def dataloader(self, batch_size):
        """Loading the images of the MNIST dataset, deviding them into batches close to 10000, not all 90000
        images are hold in the memory at once. The batched are stored in pickles.

        Args:
            path (String): path to the .gzip containing the MNIST images
            train (bool, optional): loads from the MNIST training data if True, otherwise loads from the
            testing data. Defaults to True.
        """
        image_size = 28     # MNIST images hat the size 28x28
        
        img_file = gzip.open(self.image_root, "r")
        img_file.read(16)       # first 16 bytes are non-image information
        
        label_file = gzip.open(self.label_root, "r")
        label_file.read(8)      # first 8 bytes are non-label information
        
        
        if self.train:
            num_images = 60000  # 60000 train img available in the MNIST-dataset
        else:
            num_images = 10000  # 10000 test img available

        iterations = int(num_images / batch_size)

        for i in range(iterations):
            try:
                # decode the images from bytes
                img_buf = img_file.read(image_size * image_size * batch_size)
                img_data = np.frombuffer(img_buf, dtype=np.uint8).astype(np.int32)
                img_data = img_data.reshape(batch_size, image_size, image_size, 1)
                
                # decode the labels from bytes
                label_buf = label_file.read(batch_size)
                label_data = np.frombuffer(label_buf, dtype=np.uint8).astype(np.int32)
                
            except ValueError:
                # not enough images left to fit the batch size -> cut them off and restart
                print(f"img_buf: {len(img_data)}, label_buf = {len(label_buf)}, iteration = {i}")
            
            yield img_data, label_data

