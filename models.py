import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_width=128, num_hidden_layers=2):
        super().__init__()

        if num_hidden_layers == 0:
            self.network = nn.Linear(input_dim, output_dim)
        else:
            layers = []
            layers.append(nn.Linear(input_dim, hidden_width))
            layers.append(nn.ReLU())
            for _ in range(num_hidden_layers - 1):
                layers.append(nn.Linear(hidden_width, hidden_width))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_width, output_dim))
            self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x.view(x.size(0), -1))
