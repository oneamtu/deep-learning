import torch
import numpy as np

from .models import Detector, save_model
from .utils import load_detection_data, DetectionSuperTuxDataset
from . import dense_transforms
import torch.utils.tensorboard as tb

from grader.tests import PR, point_close, box_iou

# implement training
# implement BCE
# implement test grader check for accuracy
# implement pos_weight
# focal loss?

def train(args):
    from os import path

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('device = ', device)

    model = Detector().to(device)

    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=15)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=15)

    training_data = load_detection_data('dense_data/train', 
                                        batch_size=args.batch_size,
                                        transform=dense_transforms.Compose([
                                            dense_transforms.RandomHorizontalFlip(),
                                            dense_transforms.ColorJitter(0.9, 0.9, 0.9, 0.1),
                                            dense_transforms.ToTensor(),
                                            dense_transforms.ToHeatmap()
                                        ]))
    valid_data = DetectionSuperTuxDataset('dense_data/valid', min_size=0)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=10)

    max_valid_accuracy = 0
    worse_epochs = 0

    for epoch in range(int(args.epochs)):
        # enable train mode
        model.train()

        print(f"CUDA Memory used beginning of epoch: {torch.cuda.memory_allocated()}")

        total_loss = 0.
        global_step = 0

        # train_confusion_matrix = ConfusionMatrix()
        # class_weights = 1. / torch.tensor(DENSE_CLASS_DISTRIBUTION).to(device)

        for i, (train_image, train_peaks, _train_sizes) in enumerate(training_data):
            # if i == 5:
            #     break
            train_image, train_peaks, _train_sizes = train_image.to(device), train_peaks.to(device), _train_sizes.to(device)

            y_hat = model.forward(train_image)
            # NOTE: adjust weights?
            loss = torch.nn.BCEWithLogitsLoss().forward(y_hat, train_peaks)
            # loss = torch.nn.CrossEntropyLoss(weight=class_weights).forward(y_hat, train_peaks)

            global_step = epoch*len(training_data) + i
            # train_logger.add_scalar('loss', loss, global_step)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.cpu().detach().item()
            # train_confusion_matrix.add(y_hat.argmax(1), train_peaks)

            if i < 5:
                log(train_logger, train_image[0], train_peaks[0], y_hat[0], global_step)

        # train_accuracy = train_confusion_matrix.global_accuracy.cpu().detach().item()
        train_logger.add_scalar('total_loss', total_loss, global_step)
        # train_logger.add_scalar('accuracy', train_accuracy, global_step)
        # train_logger.add_scalar('avg_accuracy', train_confusion_matrix.average_accuracy, global_step)
        # train_logger.add_scalar('iou', train_confusion_matrix.iou, global_step)

        # enable eval mode
        model.eval()

        print(f"CUDA Memory used after train: {torch.cuda.memory_allocated()}")

        # valid_confusion_matrix = ConfusionMatrix()

        pr_box = [PR() for _ in range(3)]
        pr_dist = [PR(is_close=point_close) for _ in range(3)]
        pr_iou = [PR(is_close=box_iou) for _ in range(3)]

        for i, (valid_image, *gts) in enumerate(valid_data):
            # if i == 5:
            #     break
            with torch.no_grad():
                valid_image = valid_image.to(device)
                detections = model.detect(valid_image)

                if i < 5:
                    valid_peaks, _valid_sizes = dense_transforms.detections_to_heatmap(gts, valid_image.shape[1:])
                    log(valid_logger, valid_image, valid_peaks, model.forward(valid_image).squeeze(), global_step)
        
                for i, gt in enumerate(gts):
                    pr_box[i].add(detections[i], gt)
                    pr_dist[i].add(detections[i], gt)
                    pr_iou[i].add(detections[i], gt)

        average_box_precision = np.average([pr.average_prec for pr in pr_box])
        valid_logger.add_scalars('pr_box', 
                                 { 
                                    "karts": pr_box[0].average_prec, 
                                    "bombs": pr_box[1].average_prec, 
                                    "pickup": pr_box[2].average_prec, 
                                    "average": average_box_precision
                                }, 
                                global_step)
        average_dist_precision = np.average([pr.average_prec for pr in pr_dist])
        valid_logger.add_scalars('pr_dist', 
                                 { 
                                    "karts": pr_dist[0].average_prec, 
                                    "bombs": pr_dist[1].average_prec, 
                                    "pickup": pr_dist[2].average_prec, 
                                    "average": average_dist_precision
                                }, 
                                global_step)

        train_logger.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step)
        valid_accuracy = average_box_precision + average_dist_precision
        scheduler.step(valid_accuracy)

        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {total_loss} | LR: {optimizer.param_groups[0]['lr']} " +
              f"| Valid Accuracy: {valid_accuracy}")

        if max_valid_accuracy < valid_accuracy:
            max_valid_accuracy = valid_accuracy
            save_model(model)
            worse_epochs = 0
        else:
            worse_epochs += 1
        
        if worse_epochs >= args.patience:
            print(f"Stopping at epoch {epoch}: max accuracy {max_valid_accuracy}")
            break

def log(logger, img, gt_det, det, global_step):
    """
    logger: train_logger/valid_logger
    imgs: image tensor from data loader
    gt_det: ground-truth object-center maps
    det: predicted object-center heatmaps
    global_step: iteration
    """
    logger.add_images('image', torch.tensor([img, gt_det, torch.sigmoid(det)]), global_step)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir', default='log')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=10)
    # Put custom arguments here

    args = parser.parse_args()
    train(args)
