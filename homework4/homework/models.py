import torch
import torch.nn.functional as F


def extract_peak(heatmap, max_pool_ks=7, min_score=-5, max_det=100):
    """
    Your code here.
    Extract local maxima (peaks) in a 2d heatmap.
    @heatmap: H x W heatmap containing peaks (similar to your training heatmap)
    @max_pool_ks: Only return points that are larger than a max_pool_ks x max_pool_ks window around the point
    @min_score: Only return peaks greater than min_score
    @return: List of peaks [(score, cx, cy), ...], where cx, cy are the position of a peak and score is the
             heatmap value at the peak. Return no more than max_det peaks per image
    """
    max_peaks = F.max_pool2d(heatmap[None, None], max_pool_ks, padding=max_pool_ks // 2, stride=1)

    # This is a tricky part -- we have to make sure the max of the coordinates of the peak matches
    # the value of the peak
    peak_filter = torch.eq(heatmap.view(-1), max_peaks.view(-1))
    matching_peaks = max_peaks.view(-1)[peak_filter]

    top_peaks, top_indices = torch.topk(matching_peaks, min(max_det, len(matching_peaks)))
    min_score_filter = torch.greater(top_peaks, min_score)

    peaks = top_peaks[min_score_filter]
    peak_indices = torch.arange(len(heatmap.view(-1))).to(heatmap.device)[peak_filter][top_indices[min_score_filter]]

    return [
        (peak, cx, cy)
        for peak, cx, cy in zip(
            peaks,
            torch.remainder(peak_indices, heatmap.shape[1]),
            torch.div(peak_indices, heatmap.shape[1], rounding_mode="trunc"),
        )
    ]


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


class Detector(torch.nn.Module):
    def __init__(self, layers=[16, 32, 64, 128, 256], n_input_channels=3, min_detect_score=0.02):
        """
        Your code here.
        Setup your detection network
        """
        super().__init__()
        self.mean = torch.tensor([0.2801, 0.2499, 0.2446])
        self.std_dev = torch.tensor([0.1922, 0.1739, 0.1801])

        self.first_conv = torch.nn.Conv2d(n_input_channels, layers[0], kernel_size=3, padding=1)
        self.downs = torch.nn.ModuleList([DownBlock(i, o, stride=2) for i, o in zip(layers[:-1], layers[1:])])
        self.ups = torch.nn.ModuleList([UpBlock(i, o, stride=2) for o, i in zip(layers[:-1], layers[1:])])
        self.dropout = torch.nn.Dropout()
        self.classifier = torch.nn.Conv2d(layers[0], 5, kernel_size=1)

        self.min_detect_score = min_detect_score

    def forward(self, x):
        """
        Your code here.
        Implement a forward pass through the network, use forward for training,
        and detect for detection
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
        return self.classifier(y)

    def detect(self, image):
        """
        Your code here.
        Implement object detection here.
        @image: 3 x H x W image
        @return: Three list of detections [(score, cx, cy, w/2, h/2), ...], one per class,
                 return no more than 30 detections per image per class. You only need to predict width and height
                 for extra credit. If you do not predict an object size, return w=0, h=0.
        Hint: Use extract_peak here
        Hint: Make sure to return three python lists of tuples of (float, int, int, float, float) and not a pytorch
              scalar. Otherwise pytorch might keep a computation graph in the background and your program will run
              out of memory.
        """
        return self.detections_from_heatmap(self.forward(image).squeeze(0))

    def detections_from_heatmap(self, class_heatmaps, max_pool_ks=7):
        result = []
        max_pool_sizes = F.max_pool2d(class_heatmaps[None, 3:], max_pool_ks, padding=max_pool_ks // 2, stride=1).squeeze()

        for class_heatmap in torch.sigmoid(class_heatmaps[:3]):
            peaks = extract_peak(class_heatmap, min_score=self.min_detect_score, max_pool_ks=max_pool_ks)
            if len(peaks) > 0:
                result.append(
                    torch.stack(
                        [
                            torch.tensor(
                                [score, cx, cy, max_pool_sizes[0][cy][cx], max_pool_sizes[1][cy][cx]], dtype=float
                            )
                            for score, cx, cy in peaks
                        ],
                    )
                )
            else:
                result.append(torch.empty((0, 5), dtype=float))
        return result


def save_model(model):
    from torch import save
    from os import path

    print("Saving model...")
    return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), "det.th"))


def load_model(file_name="det.th"):
    from torch import load
    from os import path

    r = Detector()
    r.load_state_dict(
        load(
            path.join(path.dirname(path.abspath(__file__)), file_name),
            map_location="cpu",
        )
    )
    return r


if __name__ == "__main__":
    """
    Shows detections of your detector
    """
    from .utils import DetectionSuperTuxDataset

    dataset = DetectionSuperTuxDataset("dense_data/valid", min_size=0)
    import torchvision.transforms.functional as TF
    from pylab import show, subplots
    import matplotlib.patches as patches

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    fig, axs = subplots(3, 4)
    model = load_model().eval().to(device)
    for i, ax in enumerate(axs.flat):
        im, kart, bomb, pickup = dataset[i]
        ax.imshow(TF.to_pil_image(im), interpolation=None)
        for k in kart:
            ax.add_patch(
                patches.Rectangle(
                    (k[0] - 0.5, k[1] - 0.5),
                    k[2] - k[0],
                    k[3] - k[1],
                    facecolor="none",
                    edgecolor="r",
                )
            )
        for k in bomb:
            ax.add_patch(
                patches.Rectangle(
                    (k[0] - 0.5, k[1] - 0.5),
                    k[2] - k[0],
                    k[3] - k[1],
                    facecolor="none",
                    edgecolor="g",
                )
            )
        for k in pickup:
            ax.add_patch(
                patches.Rectangle(
                    (k[0] - 0.5, k[1] - 0.5),
                    k[2] - k[0],
                    k[3] - k[1],
                    facecolor="none",
                    edgecolor="b",
                )
            )
        detections = model.detect(im.to(device))
        for c in range(3):
            for s, cx, cy, w, h in detections[c]:
                ax.add_patch(patches.Circle((cx, cy), radius=max(2 + s / 2, 0.1), color="rgb"[c]))
        ax.axis("off")
    show()
