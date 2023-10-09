import torch
from torchvision.transforms import functional as F

import numpy as np

import pdb

class DownBlock(torch.nn.Module):
    def __init__(self, n_input, n_output, stride=1):
        """
        2 layers of Conv2d, ReLU, first with stride
        """
        super().__init__()
        self.block = torch.nn.Sequential(
            torch.nn.Conv2d(n_input, n_output, kernel_size=3, padding=1, stride=stride, bias=False),
            torch.nn.BatchNorm2d(n_output),
            torch.nn.ReLU(),
            torch.nn.Conv2d(n_output, n_output, kernel_size=3, padding=1, stride=1, bias=False),
            torch.nn.BatchNorm2d(n_output),
            torch.nn.ReLU()
        )
        self.downsample = None
        if stride != 1 or n_input != n_output:
            self.downsample = torch.nn.Sequential(torch.nn.Conv2d(n_input, n_output, 1, stride=stride),
                                                    torch.nn.BatchNorm2d(n_output))

    def forward(self, x):
        residual_x = x
        if self.downsample is not None:
            residual_x = self.downsample(x)
        return self.block(x) + residual_x

class UpBlock(torch.nn.Module):
    def __init__(self, n_input, n_output, stride=1):
        """
        2 layers of Conv2d, ReLU, first with stride
        """
        super().__init__()
        self.up_block = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(n_input, n_output, kernel_size=3, padding=1, stride=stride, output_padding=stride-1, bias=False),
            torch.nn.BatchNorm2d(n_output),
            torch.nn.ReLU()
        )
        self.concat_block = torch.nn.Sequential(
            torch.nn.Conv2d(2*n_output, n_output, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(n_output),
            torch.nn.ReLU()
        )
        self.upsample = None
        if stride != 1 or n_input != n_output:
            self.upsample = torch.nn.Sequential(torch.nn.ConvTranspose2d(n_input, n_output, 1, stride=stride, output_padding=stride-1),
                                                    torch.nn.BatchNorm2d(n_output))

    def forward(self, x, skip_x):
        residual_x = x
        if self.upsample is not None:
            residual_x = self.upsample(x)
        up_x = self.up_block(x)
        return self.concat_block(torch.cat((up_x, skip_x), dim=1)) + residual_x

class CNNClassifier(torch.nn.Module):
    def __init__(self, layers = [32, 64, 128, 256], n_input_channels = 3):
        """
        Using blocks of Conv2d
        """
        super().__init__()
        self.first_conv = torch.nn.Conv2d(n_input_channels, layers[0], kernel_size=3, padding=1)
        self.network = torch.nn.Sequential(*[
            DownBlock(i, o, stride=2) for i, o in zip(layers[:-1], layers[1:])
        ])
        self.dropout = torch.nn.Dropout()
        self.classifier = torch.nn.Linear(layers[-1], 6)

    def forward(self, x):
        """
        @x: torch.Tensor((B,3,64,64))
        @return: torch.Tensor((B,6))
        """
        normal_x = F.normalize(x, [0.3320, 0.3219, 0.3267], [0.1922, 0.1739, 0.1801])
        y = self.first_conv(normal_x)
        y = self.network(y)
        # Global average pooling
        y = y.mean(dim=[2,3])
        y = self.dropout(y)
        return self.classifier(y)

class FCN(torch.nn.Module):
    def __init__(self, layers = [16, 32, 64, 128, 256], n_input_channels = 3):
        """
        Your code here.
        Hint: The FCN can be a bit smaller the the CNNClassifier since you need to run it at a higher resolution
        Hint: Use up-convolutions
        Hint: Use skip connections
        Hint: Use residual connections
        Hint: Always pad by kernel_size / 2, use an odd kernel_size
        """
        super().__init__()
        self.first_conv = torch.nn.Conv2d(n_input_channels, layers[0], kernel_size=3, padding=1)
        self.downs = torch.nn.ModuleList([
            DownBlock(i, o, stride=2) for i, o in zip(layers[:-1], layers[1:])
        ])
        self.ups = torch.nn.ModuleList([
            UpBlock(i, o, stride=2) for o, i in zip(layers[:-1], layers[1:])
        ])
        self.dropout = torch.nn.Dropout()
        self.classifier = torch.nn.Conv2d(layers[0], 5, kernel_size=1)

    def forward(self, x):
        """
        Your code here
        @x: torch.Tensor((B,3,H,W))
        @return: torch.Tensor((B,5,H,W))
        Hint: Apply input normalization inside the network, to make sure it is applied in the grader
        Hint: Input and output resolutions need to match, use output_padding in up-convolutions, crop the output
              if required (use z = z[:, :, :H, :W], where H and W are the height and width of a corresponding strided
              convolution
        """
        max_scale_layers = int(np.log2(min(x.shape[-1], x.shape[-2])))
        skip_ys = []

        normal_x = F.normalize(x, [0.2801, 0.2499, 0.2446], [0.1922, 0.1739, 0.1801])
        y = self.first_conv(normal_x)

        for down in self.downs[:max_scale_layers]:
            skip_ys.append(y)
            y = down(y)
        
        for up in reversed(self.ups[:max_scale_layers]):
            skip_y = skip_ys.pop()
            y = up(y, skip_y)

        y = self.dropout(y)
        return self.classifier(y)


model_factory = {
    'cnn': CNNClassifier,
    'fcn': FCN,
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
