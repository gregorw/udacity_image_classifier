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
            processed = processed.to('cuda')
        
        model = self.checkpoint.model.network
        model.eval()
        with torch.no_grad():
            # print('model: {}'.format(next(model.parameters()).device))
            # print('image: {}'.format(processed.device))
            output = self.checkpoint.model.network.forward(processed)

        probs = torch.exp(output)
        probs, indices = torch.topk(probs, 5)
        probs = probs.detach().cpu().numpy().tolist()[0]
        indices = indices.detach().cpu().numpy().tolist()[0]

        #mapping indices to classes
        idx_to_class = { v: k for k, v in self.checkpoint.class_to_idx.items() }
        classes = [ idx_to_class[i] for i in indices ]

        prediction = classes[0]
        if hasattr(self, 'cat_to_name'):
            prediction = self.cat_to_name[prediction]

        return prediction, probs[0]
