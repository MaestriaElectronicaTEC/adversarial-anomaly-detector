from abc import ABC, abstractmethod

class AbstractModel(ABC):

    @abstractmethod
    def load(self, model_dirs):
        pass

    @abstractmethod
    def preprocessing(self, datadir, data_batch_size):
        pass

    @abstractmethod
    def train(self, n_epochs, n_batch):
        pass

    @abstractmethod
    def get_metrics(self):
        pass

    @abstractmethod
    def plot(self):
        pass
