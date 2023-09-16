from .models import CNNClassifier, save_model
from .utils import accuracy, load_data
import torch
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print('device = ', device)
import torch.utils.tensorboard as tb

from torch import optim

import pdb

def train(args):
    """
    Training loop for CNNClassifier.
    """
    from os import path
    model = CNNClassifier()
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'))
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'))

    training_data = load_data('data/train')
    validation_data = load_data('data/valid')

    optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999))

    prev_loss = 0

    for epoch in range(int(args.epochs)):
        # enable train mode
        model.train()

        total_loss = 0.
        train_accuracy = 0.

        for i, (train_features, train_labels) in enumerate(training_data):
            y_hat = model.forward(train_features)
            loss = torch.nn.CrossEntropyLoss().forward(y_hat, train_labels)
            total_loss += loss

            train_logger.add_scalar('train/loss', loss, epoch*len(training_data) + i)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.detach().item()
            train_accuracy += y_hat.detach().argmax(dim=1).eq(train_labels).float().mean()

        train_logger.add_scalar('train/total_loss', total_loss, epoch*len(training_data) + i)
        train_logger.add_scalar('train/accuracy', train_accuracy / i, epoch*len(training_data) + i)

        # enable eval mode
        model.eval()

        total_validation_loss = 0.
        validation_accuracy = 0.

        for validation_features, validation_labels in validation_data:
            y_hat = model.forward(validation_features)
            validation_loss = torch.nn.CrossEntropyLoss().forward(y_hat, validation_labels)

            total_validation_loss += validation_loss
            # validation_accuracy += y_hat.detach().argmax(dim=1).eq(train_labels).float().mean()
            validation_accuracy += accuracy(y_hat, validation_labels)
        
        valid_logger.add_scalar('validation/total_loss', total_loss, epoch*len(validation_data) + i)
        valid_logger.add_scalar('validation/accuracy', validation_accuracy / i, epoch*len(validation_data) + i)
        # Early stopping - needs patience
        # if total_validation_loss > prev_loss and prev_loss > 0:
        #    break;
        prev_loss = total_validation_loss

    save_model(model)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir')
    parser.add_argument('--epochs', type=int, default=100)
    # Put custom arguments here

    args = parser.parse_args()
    train(args)
