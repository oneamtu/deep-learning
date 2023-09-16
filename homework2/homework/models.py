import torch

class CNNClassifier(torch.nn.Module):
    class Block(torch.nn.Module):
        def __init__(self, n_input, n_output, stride=1):
            """
            2 layers of Conv2d
            """
            super().__init__()
            self.block = torch.nn.Sequential(
              torch.nn.Conv2d(n_input, n_output, kernel_size=3, padding=1, stride=stride),
              torch.nn.ReLU(),
              torch.nn.Conv2d(n_output, n_output, kernel_size=3, padding=1),
              torch.nn.ReLU()
            )

        def forward(self, x):
            return self.block(x)

    def __init__(self, layers = [32, 64, 128], n_input_channels = 3):
        """
        Using blocks of Conv2d
        """
        super().__init__()
        self.network = torch.nn.Sequential(*[
            self.Block(i, o, stride=2) for i, o in zip([n_input_channels, *layers[:-1]], layers)
        ])
        self.classifier = torch.nn.Linear(layers[-1], 6)

    def forward(self, x):
        """
        @x: torch.Tensor((B,3,64,64))
        @return: torch.Tensor((B,6))
        """
        y = self.network(x)
        # Global average pooling
        y = y.mean(dim=[2,3])
        return self.classifier(y)[:,0]


def save_model(model):
    from torch import save
    from os import path
    if isinstance(model, CNNClassifier):
        return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'cnn.th'))
    raise ValueError("model type '%s' not supported!"%str(type(model)))


def load_model():
    from torch import load
    from os import path
    r = CNNClassifier()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), 'cnn.th'), map_location='cpu'))
    return r
