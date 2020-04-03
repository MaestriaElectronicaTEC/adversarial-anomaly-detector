from abc import ABC, abstractmethod

class AbstractModel:

    @abstractmethod
    def load(self, model_dirs):
        pass

    @abstractmethod
    def preprocessing(self, datadir, data_batch_size):
        pass

    @abstractmethod
    def train(self, n_epochs, n_batch):
        pass
