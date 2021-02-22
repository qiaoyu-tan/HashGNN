import numpy as np

class BatchedIterator(object):

    def __init__(self, X, batch_size, shuffle=False, epochs=-1):
        self.shuffle = shuffle
        self.epochs = epochs
        self.batch_size = batch_size
        self.X = X
        self.idx = None
        self._cur_batch = 0
        self._n_batches = (len(X) - 1) // batch_size + 1
        self._cur_epoch = 0
        self.next_epoch()

    def next_epoch(self):
        if self.shuffle:
            self.idx = np.arange(len(self.X))
            self.idx = np.random.permutation(self.idx)
        self._cur_batch = 0

    def _next(self):
        if self._cur_batch >= self._n_batches:
            self.next_epoch()
            raise StopIteration
            # if self.epochs is None or self.epochs == -1 or self._cur_epoch < self.epochs:
            #     self.next_epoch()
            # else:
            #     raise StopIteration
        start = self._cur_batch * self.batch_size
        end = start + self.batch_size
        if self.idx is not None:
            batch = [self.X[i] for i in self.idx[start:end]]
        else:
            batch = self.X[start:end]
        self._cur_batch += 1
        return batch

    def get(self):
        return self._next()