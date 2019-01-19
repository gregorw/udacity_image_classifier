class Checkpoint(object):
    def __init__(self, args=None):
        print(args)
        
    def save(self, dir):
        print('saving checkpoint')
