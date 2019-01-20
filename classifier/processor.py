import PIL
import numpy as np
import torch

class Processor:
    def __init__(self, image):
        self.image = image
        processed = self.resize(image)
        processed = self.crop(processed)
        processed = self.normalize(processed)
        processed = self.transpose(processed)
        tensor = torch.from_numpy(processed)
        tensor = tensor.type(torch.FloatTensor)
        tensor = tensor.unsqueeze_(0)
        self.result = tensor

    @staticmethod
    def resize(image):
        width, height = image.size
        aspect_ratio = width / height
        # print('ar: {}'.format(aspect_ratio))

        if width > height:
            target_height = 256
            target_width = int(target_height * aspect_ratio)
        else:
            target_width = 256
            target_height = int(target_width / aspect_ratio)

        # print('new ar: {}'.format(target_width / target_height))
        return image.resize((target_width, target_height), PIL.Image.ANTIALIAS)

    @staticmethod
    def crop(image):
        width, height = image.size
        target = 224

        left = (width - target) / 2
        bottom = (height - target) / 2
        top = bottom + target
        right = left + target

        return image.crop((left, bottom, right, top))

    @staticmethod
    def normalize(image):
        means = [0.485, 0.456, 0.406]
        stds = [0.229, 0.224, 0.225]
        image = np.array(image) / 255
        mean = np.array(means)
        std = np.array(stds)
        image = (image - mean) / std
        return image

 
    @staticmethod
    def transpose(img):
        return img.transpose((2, 0, 1))
