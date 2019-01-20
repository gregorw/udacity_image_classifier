import torch, json
from .processor import Processor

class Predictor:
    def __init__(self, checkpoint, category_names = None, gpu = False):
        self.checkpoint = checkpoint
        self.category_names = category_names
        self.gpu = gpu
        
        if category_names:
            with open(category_names, 'r') as f:
                self.cat_to_name = json.load(f)


    def perform(self, image):
        processed = Processor(image).result

        if self.gpu:
            processed.to('cuda')
        
        self.checkpoint.model.network.eval()
        with torch.no_grad():
            output = self.checkpoint.model.network.forward(processed)

        probs = torch.exp(output)
        probs, indices = torch.topk(probs, 5)
        probs = probs.detach().numpy().tolist()[0]
        indices = indices.detach().numpy().tolist()[0]

        #mapping indices to classes
        idx_to_class = { v: k for k, v in self.checkpoint.class_to_idx.items() }
        classes = [ idx_to_class[i] for i in indices ]

        prediction = classes[0]
        if hasattr(self, 'cat_to_name'):
            prediction = self.cat_to_name[prediction]

        return prediction, probs[0]
