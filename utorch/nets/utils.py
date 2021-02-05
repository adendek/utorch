from utorch.simplegrad import Variable
import numpy as np


class DataLoader(object):
    """
    Naive implementation of the DataLoader functionality, which is dedicated to feed the model with an input data.
    It returns a random sample of a data and a target. The outpu is a tuple of two simplegrad.Variables of size (batch_size, n_features) and (batch_size, 1).
    In the future this function should be changed into something more complicated.  Right now, it is implemented just to verify the correctness of the entire framework.
    Keep in mind that this function does not check the correctnes (dim) of the data.
    """

    def __init__(self, data_x, data_y, batch_size, shuffle=True):
        def get_entries(array):
            if isinstance(array, Variable):
                return array.shape()[0]
            return array.shape[0]

        if not shuffle:
            raise NotImplementedError(
                "not shuffling dataloader is not implemented yet, for the time being, please set shuffle to True")

        self.batch_size = batch_size
        self.data_x = data_x
        self.data_y = data_y
        self.n_entries = get_entries(data_x)
        self.max_batches = self.n_entries // self.batch_size
        self.iteration = 0

    def __next__(self):
        if self.iteration > self.max_batches:
            print("stop")
            raise StopIteration()
        self.iteration += 1
        batch_index = np.random.randint(self.max_batches)
        starting_row = batch_index * self.batch_size

        return (Variable(self.data_x[starting_row:starting_row + self.batch_size, :]),
                Variable(self.data_y[starting_row:starting_row + self.batch_size])
                )

    def __iter__(self):
        self.iteration = 0
        return self

    def __len__(self):
        return self.max_batches