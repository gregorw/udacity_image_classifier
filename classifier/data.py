from torchvision import datasets, transforms
import torch

class Data():
    def __init__(self, path, batch_size=20, debug=False):
        means = [0.485, 0.456, 0.406]
        stds = [0.229, 0.224, 0.225]
        standard_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(means, stds)
        ])

        self.set = datasets.ImageFolder(path, transform=standard_transforms)
        self.loader = torch.utils.data.DataLoader(self.set, batch_size=batch_size, shuffle=True)
        self.size = len(self.set.class_to_idx)
        print('Data set size: {}'.format(self.size)) if debug else None
