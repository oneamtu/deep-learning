from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.io import read_image

import csv

LABEL_NAMES = ['background', 'kart', 'pickup', 'nitro', 'bomb', 'projectile']

class SuperTuxDataset(Dataset):
    def __init__(self, dataset_path):
        """
        Your code here
        Hint: Use the python csv library to parse labels.csv

        WARNING: Do not perform data normalization here. 
        """
        self.dataset_path = dataset_path

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
        image = read_image(f'{self.dataset_path}/{self.items[idx][0]}')
        float_image = transforms.functional.convert_image_dtype(image, torch.float32)
        return (float_image, self.items[idx][1])

def load_data(dataset_path, num_workers=0, batch_size=128):
    dataset = SuperTuxDataset(dataset_path)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=False)


def accuracy(outputs, labels):
    outputs_idx = outputs.max(1)[1].type_as(labels)
    return outputs_idx.eq(labels).float().mean()
