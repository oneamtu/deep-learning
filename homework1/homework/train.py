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
    validation_data = load_data('data/valid')

    optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999))

    prev_loss = 0

    for epoch in range(int(args.epochs)):
        # enable train mode
        model.train()

        total_loss = 0.0

        for train_features, train_labels in training_data:
            result = model.forward(train_features)
            loss = ClassificationLoss().forward(result, train_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss

        print("Total training loss on epoch %i: %f" % (epoch, total_loss))

        # enable eval mode
        model.eval()

        total_validation_loss = 0.0

        for validation_features, validation_labels in validation_data:
            result = model.forward(validation_features)
            validation_loss = ClassificationLoss().forward(result, validation_labels)

            total_validation_loss += validation_loss
        
        print("Total validation loss on epoch %i: %f" % (epoch, total_validation_loss))
        # Early stopping - needs patience
        # if total_validation_loss > prev_loss and prev_loss > 0:
        #    break;
        prev_loss = total_validation_loss

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
