import torch
from torch import optim

import numpy as np

from .models import FCN, save_model
from .utils import load_dense_data, DENSE_CLASS_DISTRIBUTION, ConfusionMatrix, accuracy
from . import dense_transforms
import torch.utils.tensorboard as tb

import pdb

def train(args):
    from os import path

    device = torch.device('cuda') if args.cuda and torch.cuda.is_available() else torch.device('cpu')
    print('device = ', device)

    model = FCN().to(device)
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)

    training_data = load_dense_data('dense_data/train', batch_size=args.batch_size,
                              random_augment=True)
    validation_data = load_dense_data('dense_data/valid')

    optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)

    max_validation_accuracy = 0
    worse_epochs = 0

    for epoch in range(int(args.epochs)):
        # enable train mode
        model.train()

        print(f"CUDA Memory used beginning of epoch: {torch.cuda.memory_allocated()}")

        total_loss = 0.
        global_step = 0

        train_confusion_matrix = ConfusionMatrix()

        for i, (train_features, train_labels) in enumerate(training_data):
            train_features, train_labels = train_features.to(device), train_labels.to(device)

            y_hat = model.forward(train_features)
            # NOTE: adjust weights?
            loss = torch.nn.CrossEntropyLoss().forward(y_hat, train_labels)

            global_step = epoch*len(training_data) + i
            train_logger.add_scalar('loss', loss, global_step)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.cpu().detach().item()
            train_confusion_matrix.add(y_hat.argmax(1), train_labels)

            if i == 0:
                log(train_logger, train_features, train_labels, y_hat, global_step)

        train_accuracy = train_confusion_matrix.global_accuracy().cpu().detach().item()
        train_logger.add_scalar('total_loss', total_loss, global_step)
        train_logger.add_scalar('accuracy', train_accuracy, global_step)
        train_logger.add_scalar('avg_accuracy', train_confusion_matrix.average_accuracy(), global_step)
        train_logger.add_scalar('iou', train_confusion_matrix.iou(), global_step)

        # enable eval mode
        model.eval()

        print(f"CUDA Memory used after train: {torch.cuda.memory_allocated()}")

        valid_confusion_matrix = ConfusionMatrix()
        logged_valid_image = False

        for validation_features, validation_labels in validation_data:
            validation_features, validation_labels = validation_features.to(device), validation_labels.to(device)
            y_hat = model.forward(validation_features)

            valid_confusion_matrix.add(y_hat.argmax(1), validation_labels)

            if logged_valid_image:
                log(valid_logger, validation_features, validation_labels, y_hat, global_step)
                logged_valid_image = True
        
        validation_accuracy = valid_confusion_matrix.global_accuracy().cpu().detach().item()
        valid_logger.add_scalar('accuracy', validation_accuracy, global_step)
        valid_logger.add_scalar('avg_accuracy', valid_confusion_matrix.average_accuracy(), global_step)
        valid_logger.add_scalar('iou', valid_confusion_matrix.iou(), global_step)

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


def log(logger, imgs, lbls, logits, global_step):
    """
    logger: train_logger/valid_logger
    imgs: image tensor from data loader
    lbls: semantic label tensor
    logits: predicted logits tensor
    global_step: iteration
    """
    logger.add_image('image', imgs[0], global_step)
    logger.add_image('label', np.array(dense_transforms.label_to_pil_image(lbls[0].cpu()).
                                             convert('RGB')), global_step, dataformats='HWC')
    logger.add_image('prediction', np.array(dense_transforms.
                                                  label_to_pil_image(logits[0].argmax(dim=0).cpu()).
                                                  convert('RGB')), global_step, dataformats='HWC')

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