import torch
import numpy as np

import timeit

from .models import Detector, save_model, load_model
from .utils import load_detection_data, DetectionSuperTuxDataset
from . import dense_transforms
import torch.utils.tensorboard as tb

from grader.tests import PR, point_close, box_iou

from torch.profiler import profile, record_function, ProfilerActivity

# foca loss?
# !python3 -m homework.train --log_dir log --batch_size 256 --epochs 200 --weight_decay=1e-4 --patience 6 --loss=bce --pretrained=True

# https://github.com/clcarwin/focal_loss_pytorch/blob/e11e75bad957aecf641db6998a1016204722c1bb/focalloss.py#L6
# https://arxiv.org/pdf/1708.02002v2.pdf
# MIT license

from torch.autograd import Variable


class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        logpt = torch.nn.functional.log_softmax(input)
        logpt = logpt + target
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


def train(args, profiler=None):
    from os import path

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("device = ", device)

    if args.pretrained:
        model = load_model("best_det.th").to(device)
    else:
        model = Detector(min_detect_score=args.min_detect_score).to(device)

    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, "train"), flush_secs=15)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, "valid"), flush_secs=15)

    if args.intense_augment:
        jitter = dense_transforms.ColorJitter(0.9, 0.9, 0.9, 0.1)
    else:
        jitter = dense_transforms.ColorJitter(0.3, 0.3, 0.3, 0.1)

    training_data = load_detection_data(
        "dense_data/train",
        batch_size=args.batch_size,
        transform=dense_transforms.Compose(
            [dense_transforms.RandomHorizontalFlip(), jitter, dense_transforms.ToTensor(), dense_transforms.ToHeatmap()]
        ),
    )
    if args.test_run:
        valid_data = DetectionSuperTuxDataset("dense_data/train", min_size=0)
    else:
        valid_data = DetectionSuperTuxDataset("dense_data/valid", min_size=0)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "max", patience=3)

    max_valid_accuracy = 0
    worse_epochs = 0

    for epoch in range(int(args.epochs)):
        start_time = timeit.default_timer()
        # enable train mode
        model.train()

        print(f"CUDA Memory used beginning of epoch: {torch.cuda.memory_allocated()}")

        total_loss = 0.0
        global_step = 0

        # train_confusion_matrix = ConfusionMatrix()
        # class_weights = 1. / torch.tensor(DENSE_CLASS_DISTRIBUTION).to(device)

        for i, (train_images, train_peaks, train_sizes) in enumerate(training_data):
            if args.test_run and i == 5:
                break
            train_images, train_peaks, train_sizes = (
                train_images.to(device),
                train_peaks.to(device),
                train_sizes.to(device),
            )

            y_hat = model.forward(train_images)
            predicted_peaks, predicted_sizes = y_hat[:, :3], y_hat[:, 3:]

            if args.loss == "bce":
                positives = torch.sum(train_peaks, dim=(2, 3))
                negatives = torch.tensor([train_peaks.shape[2] * train_peaks.shape[3]]).to(device) - positives
                pos_weight = (negatives / (positives + 1e-4)).unsqueeze(-1).unsqueeze(-1)

                peak_loss = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight).forward(predicted_peaks, train_peaks)
            else:
                peak_loss = FocalLoss(gamma=args.focal_gamma)(predicted_peaks, train_peaks)

            peak_mask = torch.sum(train_peaks, dim=1, keepdim=True) > args.peak_threshold
            predicted_sizes = predicted_sizes * torch.cat((peak_mask, peak_mask), dim=1).float()
            size_loss = torch.nn.L1Loss()(predicted_sizes, train_sizes)

            loss = peak_loss * args.peak_weight + size_loss * (1 - args.peak_weight)

            global_step = epoch * len(training_data) + i
            train_logger.add_scalars("loss", {"peak": peak_loss, "size": size_loss, "sum": loss}, global_step)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.cpu().detach().item()

            if i < 5:
                log(
                    train_logger,
                    train_images[0],
                    train_peaks[0],
                    predicted_peaks[0],
                    train_sizes[0],
                    predicted_sizes[0],
                    global_step,
                )

        train_logger.add_scalar("total_loss", total_loss, global_step)

        # enable eval mode
        model.eval()

        print(
            f"CUDA Memory used after train: {torch.cuda.memory_allocated()} | Time {(timeit.default_timer() - start_time):.2f}s"
        )

        # valid_confusion_matrix = ConfusionMatrix()

        pr_box = [PR() for _ in range(3)]
        pr_dist = [PR(is_close=point_close) for _ in range(3)]
        pr_iou = [PR(is_close=box_iou) for _ in range(3)]

        prs = {
            "pr_box": pr_box,
            "pr_dist": pr_dist,
            "pr_iou": pr_iou,
        }

        if epoch % 5 == 4 or args.test_run:
            for i, (valid_image, *gts) in enumerate(valid_data):
                if args.test_run and i == 5:
                    break

                with torch.no_grad():
                    valid_image = valid_image.to(device)
                    y_hat = model(valid_image)
                    detections = model.detections_from_heatmap(y_hat.squeeze(0))

                    if i < 5:
                        # import pdb
                        # pdb.set_trace()
                        predicted_peaks, predicted_sizes = y_hat[:, :3], y_hat[:, 3:]
                        valid_peaks, valid_sizes = dense_transforms.detections_to_heatmap(
                            gts, valid_image.shape[1:], device=device
                        )
                        log(
                            valid_logger,
                            valid_image.squeeze(),
                            valid_peaks,
                            predicted_peaks[0],
                            valid_sizes,
                            predicted_sizes[0],
                            global_step + i,
                        )

                    for j, gt in enumerate(gts):
                        for pr in prs.values():
                            pr[j].add(detections[j], gt)

            valid_accuracy = 0.0
            for pr_label, pr in prs.items():
                average_precision = np.average([p.average_prec for p in pr])
                valid_accuracy += average_precision

                valid_logger.add_scalars(
                    pr_label,
                    {
                        "karts": pr[0].average_prec,
                        "bombs": pr[1].average_prec,
                        "pickup": pr[2].average_prec,
                        "average": average_precision,
                    },
                    global_step,
                )

            train_logger.add_scalar("lr", optimizer.param_groups[0]["lr"], global_step)
            scheduler.step(valid_accuracy)

            print(
                f"Epoch {epoch+1}/{args.epochs} | Train Loss: {total_loss} | LR: {optimizer.param_groups[0]['lr']} "
                + f"| Valid Accuracy: {valid_accuracy} | Time {(timeit.default_timer() - start_time):.2f}s"
            )

            if max_valid_accuracy < valid_accuracy:
                max_valid_accuracy = valid_accuracy
                save_model(model)
                worse_epochs = 0
            else:
                worse_epochs += 1

            if worse_epochs >= args.patience:
                print(f"Stopping at epoch {epoch}: max accuracy {max_valid_accuracy}")
                break


def log(logger, img, gt_det, det, gt_sizes, sizes, global_step):
    """
    logger: train_logger/valid_logger
    imgs: image tensor from data loader
    gt_det: ground-truth object-center maps
    det: predicted object-center heatmaps
    global_step: iteration
    """
    zeros = torch.zeros((1, *img.shape[1:]), device=img.device)
    images = [
        img,
        gt_det,
        torch.sigmoid(det),
        torch.sigmoid(torch.cat((gt_sizes, zeros), dim=0)),
        torch.sigmoid(torch.cat((sizes, zeros), dim=0)),
    ]
    logger.add_images("image", torch.stack(images), global_step)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--log_dir", default="log")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--intense_augment", type=bool, default=True)
    parser.add_argument("--test_run", type=bool, default=False)
    parser.add_argument("--loss", default="focal")
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--profile", type=bool, default=False)
    parser.add_argument("--min_detect_score", type=float, default=0.02)
    parser.add_argument("--peak_threshold", type=float, default=1e-4)
    parser.add_argument("--peak_weight", type=float, default=0.3)
    parser.add_argument("--pretrained", type=bool, default=False)
    # Put custom arguments here

    args = parser.parse_args()

    if args.profile:
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_stack=True) as prof:
            train(args, prof)

            print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=50))
            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=50))
            print(prof.key_averages(group_by_stack_n=5).table(sort_by="self_cuda_time_total", row_limit=5))
            prof.export_stacks("cuda_profiler_stacks.txt", "self_cuda_time_total")
            prof.export_stacks("cpu_profiler_stacks.txt", "self_cpu_time_total")
    else:
        train(args, None)
