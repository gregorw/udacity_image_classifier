from torchvision import models

class Model:
    def __init__(self, arch):
        self.network = eval('models.{}(pretrained=True)'.format(arch))
