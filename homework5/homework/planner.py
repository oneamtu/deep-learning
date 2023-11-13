import torch
import torch.nn.functional as F


def spatial_argmax(logit):
    """
    Compute the soft-argmax of a heatmap
    :param logit: A tensor of size BS x H x W
    :return: A tensor of size BS x 2 the soft-argmax in normalized coordinates (-1 .. 1)
    """
    weights = F.softmax(logit.view(logit.size(0), -1), dim=-1).view_as(logit)
    return torch.stack(
        (
            (weights.sum(1) * torch.linspace(-1, 1, logit.size(2)).to(logit.device)[None]).sum(1),
            (weights.sum(2) * torch.linspace(-1, 1, logit.size(1)).to(logit.device)[None]).sum(1),
        ),
        1,
    )


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
            torch.nn.ReLU(),
            torch.nn.Conv2d(n_output, n_output, kernel_size=3, padding=1, stride=1, bias=False),
            torch.nn.BatchNorm2d(n_output),
            torch.nn.ReLU(),
        )
        self.downsample = None
        if stride != 1 or n_input != n_output:
            self.downsample = torch.nn.Sequential(
                torch.nn.Conv2d(n_input, n_output, 1, stride=stride),
                torch.nn.BatchNorm2d(n_output),
            )

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
        # TODO: keep bias
        # TODO: remove residual
        self.up_block = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(
                n_input,
                n_output,
                kernel_size=1,
                stride=stride,
                output_padding=stride - 1,
                bias=False,
            ),
            torch.nn.BatchNorm2d(n_output),
            torch.nn.ReLU(),
        )
        self.concat_block = torch.nn.Sequential(
            torch.nn.Conv2d(2 * n_output, n_output, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(n_output),
            torch.nn.ReLU(),
        )
        self.upsample = None
        if stride != 1 or n_input != n_output:
            self.upsample = torch.nn.Sequential(
                torch.nn.ConvTranspose2d(n_input, n_output, 1, stride=stride, output_padding=stride - 1),
                torch.nn.BatchNorm2d(n_output),
            )

    def forward(self, x, skip_x):
        residual_x = x
        if self.upsample is not None:
            residual_x = self.upsample(x)
        up_x = self.up_block(x)
        return self.concat_block(torch.cat((up_x, skip_x), dim=1)) + residual_x


class Planner(torch.nn.Module):
    def __init__(self, layers=[16, 32, 64, 128, 256], n_input_channels=3):
        """
        Your code here.
        Setup your detection network
        """
        super().__init__()
        # TODO: recalculate
        self.mean = torch.tensor([0.2801, 0.2499, 0.2446])
        self.std_dev = torch.tensor([0.1922, 0.1739, 0.1801])

        self.first_conv = torch.nn.Conv2d(n_input_channels, layers[0], kernel_size=3, padding=1)
        self.downs = torch.nn.ModuleList([DownBlock(i, o, stride=2) for i, o in zip(layers[:-1], layers[1:])])
        self.ups = torch.nn.ModuleList([UpBlock(i, o, stride=2) for o, i in zip(layers[:-1], layers[1:])])
        self.dropout = torch.nn.Dropout()
        self.classifier = torch.nn.Conv2d(layers[0], 1, kernel_size=1)

    def heatmap(self, x):
        """
        Generates the heatmap
        @img: (B,3,96,128)
        return (B,96,128)
        """
        skip_ys = []

        # TODO: stdev vs variance; also dataset may be diff
        normal_x = (x - self.mean[None, :, None, None].to(x.device)) / self.std_dev[None, :, None, None].to(x.device)
        y = self.first_conv(normal_x)

        for down in self.downs:
            skip_ys.append(y)
            y = down(y)

        for up in reversed(self.ups):
            skip_y = skip_ys.pop()
            y = up(y, skip_y)

        y = self.dropout(y)
        return self.classifier(y).squeeze(1)

    def forward(self, x):
        """
        Your code here
        Predict the aim point in image coordinate, given the supertuxkart image
        @img: (B,3,96,128)
        return (B,2)
        """
        return spatial_argmax(self.heatmap(x))


def save_model(model):
    from torch import save
    from os import path

    print("Saving model!")

    if isinstance(model, Planner):
        return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), "planner.th"))
    raise ValueError("model type '%s' not supported!" % str(type(model)))


def load_model(filename="planner.th"):
    from torch import load
    from os import path

    state_dict = load(path.join(path.dirname(path.abspath(__file__)), filename), map_location="cpu")
    new_state_dict = {k: v for k, v in state_dict.items() if "classifier" not in k}

    r = Planner()
    r.load_state_dict(new_state_dict, strict=False)
    return r


if __name__ == "__main__":
    from .controller import control
    from .utils import PyTux
    from argparse import ArgumentParser

    def test_planner(args):
        # Load model
        planner = load_model().eval()
        pytux = PyTux()
        for t in args.track:
            steps, how_far, rescues = pytux.rollout(t, control, planner=planner, max_frames=1000, verbose=args.verbose)
            print(steps, how_far, rescues)
        pytux.close()

    parser = ArgumentParser("Test the planner")
    parser.add_argument("track", nargs="+")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    test_planner(args)
