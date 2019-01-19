from .checkpoint import Checkpoint

class Training:
    def __init__(self, args=None):
        self.data_dir = args.data_dir
        self.learning_rate = args.learning_rate
        self.arch = args.arch
        self.hidden_units = args.hidden_units
        self.epochs = args.epochs
        self.gpu = args.gpu
        #self.model = eval('models.{}(pretrained=True)'.format(self.arch))

    def start(self):
        print('running…')
        return Checkpoint()

    def model(self):
        None
