import gzip
import numpy as np
import pickle


class DataLoader:
    def __init__(self, batch_size):
        self.images = None
        self.labels = None
        self.current_index = None
        self.batch_size = batch_size
        self.out_filename = "data/loader/{CLS}_{ID}.pickle"

    def load_mnist_images(self, path, train=True):
        """Loading the images of the MNIST dataset, deviding them into batches close to 10000, not all 90000
        images are hold in the memory at once. The batched are stored in pickles.

        Args:
            path (String): path to the .gzip containing the MNIST images
            train (bool, optional): loads from the MNIST training data if True, otherwise loads from the
            testing data. Defaults to True.
        """
        # choose the num of loaded images depending on the use case
        if train:
            num_images = 60000
        else:
            num_images = 10000

        with gzip.open(path, "r") as f:
            image_size = 28
            f.read(16)

            # set a buf size close to 10000 - the desired number of images to load
            buf_size = int(10000 / self.batch_size) * self.batch_size
            iterations = int(np.ceil(num_images / buf_size))

            for i in range(iterations):
                try:
                    # decode the images frome bytes
                    buf = f.read(image_size * image_size * buf_size)
                    data = np.frombuffer(buf, dtype=np.uint8).astype(np.int32)
                    data = data.reshape(buf_size, image_size, image_size, 1)
                except ValueError:
                    # less images left then buf_size --> handle the last fraction of the dataset
                    data = data.reshape(
                        int(data.shape[0] / image_size**2), image_size, image_size, 1
                    )

                # safe the loaded images in batches inside a dict
                pik = {}
                for j in range(0, buf_size, self.batch_size):
                    pik[i * buf_size + j] = data[j : j + self.batch_size]

                # choose filename to store the dict as pickle
                if train:
                    name = self.out_filename.replace("{CLS}", "train")
                else:
                    name = self.out_filename.replace("{CLS}", "test")
                name = name.replace("{ID}", str(i))
                # write the pickle
                with open(name, "wb") as file:
                    pickle.dump(pik, file, protocol=pickle.HIGHEST_PROTOCOL)

    def load_mnist_labels(self, path, train=True):
        """Loading the labels of the MNIST dataset, deviding them into batches close to 10000, not all 90000
        images are hold in the memory at once. The batched are stored in pickles.

        Args:
            path (String): path to the .gzip containing the MNIST images
            train (bool, optional): loads from the MNIST training data if True, otherwise loads from the
            testing data. Defaults to True.
        """
        # choose the num of loaded labels depending on the use case
        if train:
            num_images = 60000
        else:
            num_images = 10000

        # set a buf size close to 10000 - the desired number of labels to load
        buf_size = int(10000 / self.batch_size) * self.batch_size
        iterations = int(np.ceil(num_images / buf_size))

        with gzip.open(path, "r") as f:
            f.read(8)

            for i in range(iterations):
                buf = f.read(buf_size)
                # decode the labels frome bytes
                labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int32)
                
                # safe the loaded labels in batches inside a dict
                pik = {}
                for j in range(0, buf_size, self.batch_size):
                    pik[i * buf_size + j] = labels[j : j + self.batch_size]
                
                # choose filename to store the dict as pickle
                if train:
                    name = f"data/loader/train_labels_{i}.pickle"
                else:
                    name = f"data/loader/test_labels_{i}.pickle"
                # write the pickle
                with open(name, "wb") as file:
                    pickle.dump(pik, file, protocol=pickle.HIGHEST_PROTOCOL)

    def load_next_file(self):
        """Loads the next pickle into the local memory"""
        self.current_index += 1

        with open(f"data/loader/train_{self.current_index}.pickle", "rb") as f:
            self.images = pickle.load(f)

        with open(f"data/loader/train_labels_{self.current_index}.pickle", "rb") as f:
            self.labels = pickle.load(f)

    def get_batch(self, img_number):
        """given the index of an image in respect to the whole dataset, the corresponding batch of images will
        be returned, which starts from the given img_index 

        Args:
            img_number (int): index of the first image on a batch

        Returns:
            tuple: (X, Y) where X holds the image data and Y the labels of the batch 
        """
        # if nothing loaded -> load the first training images and labels
        if self.images is None:
            self.current_index = 0

            with open("data/loader/train_0.pickle", "rb") as f:
                self.images = pickle.load(f)

            with open("data/loader/train_labels_0.pickle", "rb") as f:
                self.labels = pickle.load(f)
                
        # return the desired batch
        if img_number in self.images:
            return self.images[img_number], self.labels[img_number]
        else:
            # if batch not in pickle - go recursively through the next pickles
            self.load_next_file()
            return self.get_batch(img_number)

    def reset(self):
        """Loads the first pickle of the data. This method should be run before the start of a new epoch"""
        self.current_index = -1
        self.load_next_file()
