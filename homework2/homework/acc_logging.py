from os import path
import torch
import torch.utils.tensorboard as tb


def test_logging(train_logger, valid_logger):

    """
    Your code here.
    Finish logging the dummy loss and accuracy
    Log the loss every iteration, the accuracy only after each epoch
    Make sure to set global_step correctly, for epoch=0, iteration=0: global_step=0
    Call the loss 'loss', and accuracy 'accuracy' (no slash or other namespace)
    """

    # This is a strongly simplified training loop
    for epoch in range(10):
        torch.manual_seed(epoch)
        dummy_train_accuracy = torch.zeros(10)
        dummy_validation_accuracy = torch.zeros(10)
        for iteration in range(20):
            dummy_train_loss = 0.9**(epoch+iteration/20.)
            dummy_train_accuracy += (epoch/10. + torch.randn(10))/20.
            train_logger.add_scalar('loss', dummy_train_loss, epoch*20 + iteration)
        train_logger.add_scalar('accuracy', dummy_train_accuracy.mean(), epoch*20 + iteration)
        torch.manual_seed(epoch)
        for iteration in range(10):
            dummy_validation_accuracy += (epoch / 10. + torch.randn(10))/20.
        valid_logger.add_scalar('accuracy', dummy_validation_accuracy.mean(), epoch*20 + iteration)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('log_dir')
    args = parser.parse_args()
    train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'))
    valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'test'))
    test_logging(train_logger, valid_logger)
