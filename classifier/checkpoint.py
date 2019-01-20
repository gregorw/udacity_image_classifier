import time, torch
from torch import nn
from .model import Model
from .classifier import Classifier

class Checkpoint:
    def __init__(self, 
                 learning_rate = None,
                 epochs = None,
                 arch = None,
                 input_size = None,
                 hidden_units = None,
                 output_size = None,
                 dropout = None,
                 device = None,
                 class_to_idx = None,
                 classifier = None):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.arch = arch
        self.input_size = input_size
        self.hidden_units = hidden_units
        self.output_size = output_size
        self.dropout = dropout
        self.device = device
        self.class_to_idx = class_to_idx
        self.classifier = classifier


    def save(self, save_dir):
        self.data = {}
        self.data['learning_rate'] = self.learning_rate
        self.data['epochs'] = self.epochs
        self.data['arch'] = self.arch
        self.data['input_size'] = self.input_size
        self.data['hidden_units'] = self.hidden_units
        self.data['output_size'] = self.output_size
        self.data['dropout'] = self.dropout
        self.data['device'] = self.device
        self.data['class_to_idx'] = self.class_to_idx
        self.data['classifier'] = self.classifier
        
        filename = '{}/{}_checkpoint.pth'.format(save_dir, int(time.time()))
        torch.save(self.data, filename)
        print('Checkpoint written to {}'.format(filename))

    
    def set_model(self, model, gpu = False):
        self.model = model

        if gpu:
            self.model.network = self.model.network.to('cuda')


    # Class method
    def load(file_name, gpu = False):
        # https://discuss.pytorch.org/t/on-a-cpu-device-how-to-load-checkpoint-saved-on-gpu-device/349
        # torch.load('my_file.pt', map_location=lambda storage, loc: storage)
        if gpu:
            checkpoint = torch.load(file_name)
        else:
            # torch_device = torch.device('cpu')
            checkpoint = torch.load(file_name, map_location='cpu')
        
        cp = Checkpoint(
            learning_rate = checkpoint['learning_rate'],
            epochs = checkpoint['epochs'],
            arch = checkpoint['arch'],
            input_size = checkpoint['input_size'],
            hidden_units = checkpoint['hidden_units'],
            output_size = checkpoint['output_size'],
            dropout = checkpoint['dropout'],
            device = checkpoint['device'],
            class_to_idx = checkpoint['class_to_idx'],
            classifier = checkpoint['classifier']
        )
        
        clf = Classifier(cp.input_size, cp.hidden_units, cp.dropout,  cp.output_size)
        clf.load_state(cp.classifier)
        
        model = Model(cp.arch)
        model.classifier(clf)
        cp.set_model(model, gpu)

        return cp
