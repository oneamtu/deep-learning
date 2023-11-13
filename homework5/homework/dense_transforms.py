# Source: https://github.com/pytorch/vision/blob/master/references/segmentation/transforms.py
import numpy as np
import random
import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F


class RandomHorizontalFlip(object):
    def __init__(self, flip_prob=0.5):
        self.flip_prob = flip_prob

    def __call__(self, image, *args):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            args = tuple(np.array([-point[0], point[1]], dtype=point.dtype) for point in args)
        return (image,) + args


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, *args):
        for t in self.transforms:
            image, *args = t(image, *args)
        return (image,) + tuple(args)


class ColorJitter(T.ColorJitter):
    def __call__(self, image, *args):
        return (super().__call__(image),) + args


class ToTensor(object):
    def __call__(self, image, *args):
        return (F.to_tensor(image),) + args


class ToHeatmap(object):
    def __init__(self, radius=2):
        self.radius = radius

    def __call__(self, image, label):
        peak = detections_to_heatmap(label, image.shape[1:], radius=self.radius, device=image.device)
        return image, label, peak


def detections_to_heatmap(label, shape, radius=2, device=None):
    with torch.no_grad():
        cx, cy = torch.from_numpy(label)
        x = torch.arange(shape[1], dtype=cx.dtype, device=device)
        y = torch.arange(shape[0], dtype=cy.dtype, device=device)
        gx = (-(((x - (1 + cx) * shape[1] / 2) / radius) ** 2)).exp()
        gy = (-(((y - (1 + cy) * shape[0] / 2) / radius) ** 2)).exp()
        peak = gx[None] * gy[:, None]
        return peak
