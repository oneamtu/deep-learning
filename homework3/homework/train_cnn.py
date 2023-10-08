from .models import CNNClassifier, save_model
from .utils import ConfusionMatrix, load_data, LABEL_NAMES, accuracy
import torch
from torch import optim
import torchvision
import torch.utils.tensorboard as tb

import numpy as np

# ~87% accuracy, 30-40 epochs

# !python3 -m homework.train_cnn --log_dir log_bn --batch_size 32 --epochs 30 --cuda True
# Input normalization - 90% accuracy, 30 epochs

# !python3 -m homework.train_cnn --log_dir log_bn_residual --batch_size 512 --epochs 30 --patience 20 --cuda True
# bigger batch not necessarily better; batch size 512 worse than 32

# !python3 -m homework.train_cnn --log_dir log_bn_residual --batch_size 64 --epochs 30 --patience 20 --cuda True
# Residual blocks - 91 max, could go for longer

# Dropout
# Data augmentations (Both geometric and color augmentations are important. Be aggressive here. Different levels of supertux have radically different lighting.)
# Weight regularization

# Early stopping -- helpful for capturing max
# LR schedule - helpful to stabilize convergence; try higher LR default to see if faster convergence.
# !python3 -m homework.train_cnn --log_dir log_bn_residual_lr_sched --batch_size 128 --epochs 60 --patience 15 --cuda True
# Stopping at epoch 34: max accuracy 0.9091517857142857
# !python3 -m homework.train_cnn --log_dir log_bn_residual_lr_sched --batch_size 128 --epochs 60 --patience 15 --cuda True
# Stopping at epoch 39: max accuracy 0.8911830357142857 start_lr: 0.01

def train(args):
    """
    Training loop for CNNClassifier.
    """
    device = torch.device('cuda') if args.cuda and torch.cuda.is_available() else torch.device('cpu')
    print('device = ', device)

    from os import path
    model = CNNClassifier().to(device)
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'))
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'))

    training_data = load_data('data/train', batch_size=args.batch_size, 
                              random_crop=(64, 64), random_horizontal_flip=True)
    validation_data = load_data('data/valid')

    optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)

    max_validation_accuracy = 0
    worse_epochs = 0

    for epoch in range(int(args.epochs)):
        # enable train mode
        model.train()

        total_loss = 0.
        train_accuracies = []

        print(f"CUDA Memory used beginning of epoch: {torch.cuda.memory_allocated()}")

        global_step = 0

        for i, (train_features, train_labels) in enumerate(training_data):
            train_features, train_labels = train_features.to(device), train_labels.to(device)

            y_hat = model.forward(train_features)
            loss = torch.nn.CrossEntropyLoss().forward(y_hat, train_labels)

            global_step = epoch*len(training_data) + i
            train_logger.add_scalar('loss', loss, global_step)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.cpu().detach().item()
            train_accuracies.append(accuracy(y_hat, train_labels).cpu().detach().item())

        train_accuracy = np.mean(train_accuracies)
        train_logger.add_scalar('total_loss', total_loss, global_step)
        train_logger.add_scalar('accuracy', train_accuracy, global_step)

        # enable eval mode
        model.eval()

        print(f"CUDA Memory used after train: {torch.cuda.memory_allocated()}")

        validation_accuracies = []

        for validation_features, validation_labels in validation_data:
            validation_features, validation_labels = validation_features.to(device), validation_labels.to(device)
            y_hat = model.forward(validation_features)

            # validation_accuracy += y_hat.detach().argmax(dim=1).eq(train_labels).float().mean()
            validation_accuracies.append(accuracy(y_hat, validation_labels).cpu().detach().item())
        
        validation_accuracy = np.mean(validation_accuracies)
        valid_logger.add_scalar('accuracy', validation_accuracy, epoch*len(training_data) + i)

        train_logger.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step)
        scheduler.step(validation_accuracy)

        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {total_loss} | LR: {optimizer.param_groups[0]['lr']} " +
              f"| Train Accuracy: {train_accuracy} | Validation Accuracy: {validation_accuracy}")

        # Early stopping - needs patience
        if max_validation_accuracy < validation_accuracy:
            max_validation_accuracy = validation_accuracy
            save_model(model)
            worse_epochs = 0
        else:
            worse_epochs += 1
        
        if worse_epochs >= args.patience:
            print(f"Stopping at epoch {epoch}: max accuracy {max_validation_accuracy}")
            break

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', default='log')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--cuda', type=bool, default=False)
    # Put custom arguments here

    args = parser.parse_args()
    train(args)
