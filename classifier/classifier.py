from torch import nn

class Classifier:
    def __init__(self, input_size, hidden_units, dropout, output_size):
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_units),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_units, output_size),
            nn.LogSoftmax(dim=1)
        )

    def load_state(self, dict):
        self.network.load_state_dict(dict)
