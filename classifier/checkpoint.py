import time, torch

class Checkpoint:
    def __init__(self, data):
        self.data = data

    def save(self, save_dir):
        torch.save(
            self.data, '{}/{}_checkpoint.pth'.format(save_dir, int(time.time()))
        )
