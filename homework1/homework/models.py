import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import IMAGE_SIZE, LABEL_NAMES
IMAGE_LINEAR_SIZE = IMAGE_SIZE[0] * IMAGE_SIZE[1] * IMAGE_SIZE[2]
MIDDLE_LINEAR_SIZE = IMAGE_SIZE[0] * IMAGE_SIZE[1]

class ClassificationLoss(torch.nn.Module):
    def forward(self, input, target):
        """
        Your code here

        Compute mean(-log(softmax(input)_label))

        @input:  torch.Tensor((B,C))
        @target: torch.Tensor((B,), dtype=torch.int64)

        @return:  torch.Tensor((,))

        Hint: Don't be too fancy, this is a one-liner
        """
        return torch.nn.CrossEntropyLoss()(input, target)


class LinearClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()

        """
        Your code here
        """
        self.model = nn.Linear(IMAGE_LINEAR_SIZE, len(LABEL_NAMES))

    def forward(self, x):
        """
        Your code here

        @x: torch.Tensor((B,3,64,64))
        @return: torch.Tensor((B,6))
        """
        return self.model.forward(x.view(-1, IMAGE_LINEAR_SIZE))


class MLPClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()

        """
        Your code here
        """
        self.model = nn.Sequential(
            nn.Linear(IMAGE_LINEAR_SIZE, MIDDLE_LINEAR_SIZE),
            nn.ReLU(),
            nn.Linear(MIDDLE_LINEAR_SIZE, len(LABEL_NAMES)))

    def forward(self, x):
        """
        Your code here

        @x: torch.Tensor((B,3,64,64))
        @return: torch.Tensor((B,6))
        """
        return self.model.forward(x.view(-1, IMAGE_LINEAR_SIZE))


model_factory = {
    'linear': LinearClassifier,
    'mlp': MLPClassifier,
}


def save_model(model):
    from torch import save
    from os import path
    for n, m in model_factory.items():
        if isinstance(model, m):
            return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), '%s.th' % n))
    raise ValueError("model type '%s' not supported!" % str(type(model)))


def load_model(model):
    from torch import load
    from os import path
    r = model_factory[model]()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), '%s.th' % model), map_location='cpu'))
    return r
