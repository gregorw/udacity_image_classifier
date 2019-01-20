import torch
from torch import nn, optim
from .checkpoint import Checkpoint
from .data import Data
from .model import Model
from .classifier import Classifier

class Training:
    def __init__(self, args=None):
        self.debug = False
        print(args)
        self.data_dir = args.data_dir
        self.arch = args.arch
        self.hidden_units = args.hidden_units
        self.learning_rate = args.learning_rate
        self.epochs = args.epochs
        self.device = 'cuda' if args.gpu else 'cpu'
        self.dropout = args.dropout
        self.data = Data(self.data_dir + '/train')
        self.validation_data = Data(self.data_dir + '/valid', debug=self.debug)

    def start(self):
        print('Training started…') if self.debug else None
        self.prepare_model()
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(self.model.network.classifier.parameters(), lr = self.learning_rate)
        print_every = 40
        steps = 0

        for e in range(self.epochs):
            print('Epoch {}/{}'.format(e+1, self.epochs)) if self.debug else None
            running_loss = 0
            for inputs, labels in self.data.loader:
                self.model.network.train()
                steps += 1
                # Move input and label tensors to the GPU
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model.network.forward(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                if steps % print_every == 0:
                    validation_loss, accuracy = self.validation(self.model.network, self.validation_data.loader, criterion, self.debug)
                    print("Epoch: {}/{}... ".format(e+1, self.epochs),
                          "Training Loss: {:.4f} ".format(running_loss/print_every),
                          "Validation Loss: {:.3f} ".format(validation_loss),
                          "Validation Accuracy: {:.3f}".format(accuracy))
                    running_loss = 0

        checkpoint = Checkpoint(
            learning_rate = self.learning_rate,
            epochs = self.epochs,
            arch = self.arch,
            input_size = self.model.input_size,
            hidden_units = self.hidden_units,
            output_size = self.data.size,
            dropout = self.dropout,
            device = self.device,
            class_to_idx = self.data.set.class_to_idx,
            classifier = self.model.network.classifier.state_dict()
        )
        return checkpoint


    def validation(self, model, dataloader, criterion, log = False):
        print('validation…') if log else None
        loss = 0
        accuracy = 0
        batch_count = len(dataloader)
        set_size = len(dataloader.dataset)
        model.eval()
        with torch.no_grad():
            for i, (images, labels) in enumerate(dataloader, start = 1):
                images, labels = images.to(self.device), labels.to(self.device)
                output = model.forward(images)
                loss += criterion(output, labels).item()
                ps = torch.exp(output)
                matches = (labels.data == ps.max(dim=1)[1])
                accuracy += matches.type(torch.FloatTensor).mean()
                if log == True:
                    print("Batch: {}/{}... ".format(i, batch_count),
                          "Validation Loss: {:.3f} ".format(loss / i),
                          "Validation Accuracy: {:.3f}".format(accuracy / i))

        return loss / batch_count, accuracy / batch_count


    def prepare_model(self):
        self.model = Model(self.arch)
        self.model.classifier(Classifier(
            self.model.input_size,
            self.hidden_units,
            self.dropout,
            self.data.size
        ))
        self.model.network.to(self.device)
