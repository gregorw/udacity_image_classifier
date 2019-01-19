from torchvision import models

class Model:
    def __init__(self, arch):
        self.network = eval('models.{}(pretrained=True)'.format(arch))
        self.in_size = self.network.classifier[0].in_features
        # Freeze parameters so we don't backprop through them
        for param in self.network.parameters():
            param.requires_grad = False

    def classifier(self, classifier):
        self.network.classifier = classifier.network
