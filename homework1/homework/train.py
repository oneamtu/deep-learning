from .models import ClassificationLoss, model_factory, save_model
from .utils import accuracy, load_data

from torch import optim
import random

def train(args):
    model = model_factory[args.model]()

    """
    Your code here

    """
    training_data = load_data('data/train')
    optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999))

    for epoch in range(args.epochs):
        # enable train mode
        model.train()

        total_loss = 0.0

        for train_features, train_labels in training_data:
            model.zero_grad()

            result = model.forward(train_features)
            loss = ClassificationLoss().forward(result, train_labels)
            loss.backward()
            optimizer.step()

            total_loss += loss

        print("Total loss on epoch %i: %f" % (epoch, total_loss))

        # enable eval mode
        model.eval()
    save_model(model)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model', choices=['linear', 'mlp'], default='linear')
    parser.add_argument('-e', '--epochs', default=5)
    parser.add_argument('-b', '--batch_size', default=16)
    # Put custom arguments here

    args = parser.parse_args()
    train(args)
