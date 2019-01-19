from torchvision import datasets, transforms, models
from .checkpoint import Checkpoint
from .data import Data
from .model import Model
from .classifier import Classifier

class Training:
    def __init__(self, args=None):
        self.data_dir = args.data_dir
        self.arch = args.arch
        self.hidden_units = args.hidden_units
        self.learning_rate = args.learning_rate
        self.epochs = args.epochs
        self.gpu = args.gpu
        self.device = 'cuda' if args.gpu else 'cpu'
        self.dropout = args.dropout
        self.data = Data(self.data_dir + '/train')
        self.validation_data = Data(self.data_dir + '/valid')

    def start(self):
        print('Training started…')
        self.prepare_model()
        return Checkpoint()

    def prepare_model(self):
        self.model = Model(self.arch)
        self.model.classifier(Classifier(
            self.model.in_size,
            self.hidden_units,
            self.dropout,
            self.data.size
        ))
