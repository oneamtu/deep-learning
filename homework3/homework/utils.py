import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as F
from torchvision.io import read_image

import csv

from . import dense_transforms

LABEL_NAMES = ['background', 'kart', 'pickup', 'nitro', 'bomb', 'projectile']
DENSE_LABEL_NAMES = ['background', 'kart', 'track', 'bomb/projectile', 'pickup/nitro']
# Distribution of classes on dense training set (background and track dominate (96%)
DENSE_CLASS_DISTRIBUTION = [0.52683655, 0.02929112, 0.4352989, 0.0044619, 0.00411153]

class SuperTuxDataset(Dataset):
    def __init__(self, dataset_path, random_augment=False):
        """
        Your code here
        Hint: Use the python csv library to parse labels.csv

        WARNING: Do not perform data normalization here. 
        """
        self.dataset_path = dataset_path
        self.transform = self.get_transform(random_augment)

        with open(f'{dataset_path}/labels.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            next(csv_reader, None) # skip headers
            self.items = [(row[0], LABEL_NAMES.index(row[1]))
                          for row in csv_reader]


    def __len__(self):
        """
        Your code here
        """
        return len(self.items)

    def __getitem__(self, idx):
        """
        Your code here
        return a tuple: img, label
        """
        raw_image = read_image(f'{self.dataset_path}/{self.items[idx][0]}')
        image = self.transform(raw_image)
        return (image, self.items[idx][1])
    
    def get_transform(self, random_augment):
        transform = []
        if random_augment:
            # transform.append(transforms.RandomResizedCrop(self.random_crop))
            transform.append(transforms.RandomHorizontalFlip())
            transform.append(transforms.ColorJitter(brightness=2., saturation=.5))
        transform.append(transforms.ConvertImageDtype(torch.float32))
        return transforms.Compose(transform)


class DenseSuperTuxDataset(Dataset):
    def __init__(self, dataset_path, transform=None, random_augment=False):
        from glob import glob
        from os import path
        self.files = []
        for im_f in glob(path.join(dataset_path, '*_im.jpg')):
            self.files.append(im_f.replace('_im.jpg', ''))
        if transform is None:
            self.transform = self.get_transform(random_augment)
        else:
            self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        b = self.files[idx]
        im = Image.open(b + '_im.jpg')
        lbl = Image.open(b + '_seg.png')
        if self.transform is not None:
            im, lbl = self.transform(im, lbl)
        return im, lbl

    def get_transform(self, random_augment):
        transform = []
        if random_augment:
            # transform.append(transforms.RandomResizedCrop(self.random_crop))
            transform.append(dense_transforms.RandomHorizontalFlip())
            transform.append(dense_transforms.ColorJitter(brightness=2., saturation=.5))
        transform.append(dense_transforms.ToTensor())
        return dense_transforms.Compose(transform)

def load_data(dataset_path, num_workers=0, batch_size=128, **kwargs):
    dataset = SuperTuxDataset(dataset_path, **kwargs)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=True)


def load_dense_data(dataset_path, num_workers=0, batch_size=32, **kwargs):
    dataset = DenseSuperTuxDataset(dataset_path, **kwargs)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=True)


def _one_hot(x, n):
    return (x.view(-1, 1) == torch.arange(n, dtype=x.dtype, device=x.device)).int()

def accuracy(outputs, labels):
    outputs_idx = outputs.max(1)[1].type_as(labels)
    return outputs_idx.eq(labels).float().mean()


class ConfusionMatrix(object):
    def _make(self, preds, labels):
        label_range = torch.arange(self.size, device=preds.device)[None, :]
        preds_one_hot, labels_one_hot = _one_hot(preds, self.size), _one_hot(labels, self.size)
        return (labels_one_hot[:, :, None] * preds_one_hot[:, None, :]).sum(dim=0).detach()

    def __init__(self, size=5):
        """
        This class builds and updates a confusion matrix.
        :param size: the number of classes to consider
        """
        self.matrix = torch.zeros(size, size)
        self.size = size

    def add(self, preds, labels):
        """
        Updates the confusion matrix using predictions `preds` (e.g. logit.argmax(1)) and ground truth `labels`
        """
        self.matrix = self.matrix.to(preds.device)
        self.matrix += self._make(preds, labels).float()

    @property
    def class_iou(self):
        true_pos = self.matrix.diagonal()
        return true_pos / (self.matrix.sum(0) + self.matrix.sum(1) - true_pos + 1e-5)

    @property
    def iou(self):
        return self.class_iou.mean()

    @property
    def global_accuracy(self):
        true_pos = self.matrix.diagonal()
        return true_pos.sum() / (self.matrix.sum() + 1e-5)

    @property
    def class_accuracy(self):
        true_pos = self.matrix.diagonal()
        return true_pos / (self.matrix.sum(1) + 1e-5)

    @property
    def average_accuracy(self):
        return self.class_accuracy.mean()

    @property
    def per_class(self):
        return self.matrix / (self.matrix.sum(1, keepdims=True) + 1e-5)

if __name__ == '__main__':
    # NOTE: code for mean/stddev was taken from this post
    # https://stackoverflow.com/questions/60101240/finding-mean-and-standard-deviation-across-image-channels-pytorch
    loader = load_data('data/train')
    nimages = 0
    mean = 0.0
    var = 0.0
    for i_batch, batch_target in enumerate(loader):
        batch = batch_target[0]
        # Rearrange batch to be the shape of [B, C, W * H]
        batch = batch.view(batch.size(0), batch.size(1), -1)
        # Update total number of images
        nimages += batch.size(0)
        # Compute mean and std here
        mean += batch.mean(2).sum(0) 
        var += batch.var(2).sum(0)

    loader = load_data('data/valid')
    for i_batch, batch_target in enumerate(loader):
        batch = batch_target[0]
        # Rearrange batch to be the shape of [B, C, W * H]
        batch = batch.view(batch.size(0), batch.size(1), -1)
        # Update total number of images
        nimages += batch.size(0)
        # Compute mean and std here
        mean += batch.mean(2).sum(0) 
        var += batch.var(2).sum(0)

    mean /= nimages
    var /= nimages
    std = torch.sqrt(var)

    print(f"Dataset mean: {mean}")
    print(f"Dataset std: {std}")

    loader = load_dense_data('dense_data/train')
    nimages = 0
    mean = 0.0
    var = 0.0
    for i_batch, batch_target in enumerate(loader):
        batch = batch_target[0]
        # Rearrange batch to be the shape of [B, C, W * H]
        batch = batch.view(batch.size(0), batch.size(1), -1)
        # Update total number of images
        nimages += batch.size(0)
        # Compute mean and std here
        mean += batch.mean(2).sum(0) 
        var += batch.var(2).sum(0)

    loader = load_dense_data('dense_data/valid')
    for i_batch, batch_target in enumerate(loader):
        batch = batch_target[0]
        # Rearrange batch to be the shape of [B, C, W * H]
        batch = batch.view(batch.size(0), batch.size(1), -1)
        # Update total number of images
        nimages += batch.size(0)
        # Compute mean and std here
        mean += batch.mean(2).sum(0) 
        var += batch.var(2).sum(0)

    mean /= nimages
    var /= nimages
    std = torch.sqrt(var)

    print(f"Dense Dataset mean: {mean}")
    print(f"Dense Dataset std: {std}")

    dataset = DenseSuperTuxDataset('dense_data/train', transform=dense_transforms.Compose(
        [dense_transforms.RandomHorizontalFlip(), dense_transforms.ToTensor()]))
    from pylab import show, imshow, subplot, axis

    for i in range(15):
        im, lbl = dataset[i]
        subplot(5, 6, 2 * i + 1)
        imshow(F.to_pil_image(im))
        axis('off')
        subplot(5, 6, 2 * i + 2)
        imshow(dense_transforms.label_to_pil_image(lbl))
        axis('off')
    show()
    import numpy as np

    c = np.zeros(5)
    for im, lbl in dataset:
        c += np.bincount(lbl.view(-1), minlength=len(DENSE_LABEL_NAMES))
    print(100 * c / np.sum(c))
