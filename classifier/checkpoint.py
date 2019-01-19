class Checkpoint:
    def __init__(self, data):
        self.data = data

    def save(self, dir = '.'):
        torch.save(
            self.data, dir + '/{}_checkpoint.pth'.format(int(time.time()))
        )
