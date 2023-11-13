from .planner import Planner, save_model, load_model, spatial_argmax
import torch
import torch.utils.tensorboard as tb
import numpy as np
from .utils import load_data
from . import dense_transforms

import timeit
from torch.profiler import profile, record_function, ProfilerActivity


def train(args):
    from os import path

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("device = ", device)

    if args.pretrained is not None:
        model = load_model(args.pretrained).to(device)
    else:
        model = Planner().to(device)

    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, "train"), flush_secs=15)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, "valid"), flush_secs=15)

    if args.intense_augment:
        jitter = dense_transforms.ColorJitter(0.9, 0.9, 0.9, 0.1)
    else:
        jitter = dense_transforms.ColorJitter(0.3, 0.3, 0.3, 0.1)

    training_data = load_data(
        "drive_data/train",
        batch_size=args.batch_size,
        transform=dense_transforms.Compose(
            [dense_transforms.RandomHorizontalFlip(), jitter, dense_transforms.ToTensor()]
        ),
    )
    valid_data = load_data(
        "drive_data/valid",
        batch_size=args.batch_size,
        transform=dense_transforms.Compose([dense_transforms.ToTensor()]),
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "max", patience=5)

    max_valid_accuracy = 0
    worse_epochs = 0

    for epoch in range(int(args.epochs)):
        start_time = timeit.default_timer()
        # enable train mode
        model.train()

        print(f"CUDA Memory used beginning of epoch: {torch.cuda.memory_allocated()}")

        total_loss = 0.0
        global_step = 0

        for i, (train_images, train_labels) in enumerate(training_data):
            if args.test_run and i == 5:
                break
            train_images, train_labels = train_images.to(device), train_labels.to(device)

            predicted_heatmaps = model.heatmap(train_images)
            predicted_labels = spatial_argmax(predicted_heatmaps)

            loss = torch.nn.MSELoss().forward(predicted_labels, train_labels)

            global_step = epoch * len(training_data) + i
            train_logger.add_scalars(
                "loss",
                {"loss": loss},
                global_step,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.cpu().detach().item()

            if i < 5:
                log(
                    train_logger,
                    train_images,
                    train_labels,
                    predicted_labels,
                    global_step,
                    pred_peaks=predicted_heatmaps,
                )

        train_logger.add_scalar("total_loss", total_loss, global_step)

        # enable eval mode
        model.eval()

        print(
            f"CUDA Memory used after train: {torch.cuda.memory_allocated()} | Time {(timeit.default_timer() - start_time):.2f}s"
        )

        valid_accuracy = 0.0

        for i, (valid_images, valid_labels) in enumerate(valid_data):
            if args.test_run and i == 5:
                break

            with torch.no_grad():
                valid_images = valid_images.to(device)
                predicted_labels = model(valid_images)

                valid_accuracy += torch.sum(
                    torch.pairwise_distance(predicted_labels, valid_labels) < 5e-2
                ).cpu().detach().item() / len(valid_labels)

                if i < 5:
                    log(
                        valid_logger,
                        valid_images,
                        valid_labels,
                        predicted_labels,
                        global_step + i,
                    )

        valid_accuracy /= i

        valid_logger.add_scalars(
            "accuracy",
            {"accuracy": valid_accuracy},
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


def log(logger, img, label, pred, global_step, pred_peaks=None):
    """
    logger: train_logger/valid_logger
    img: image tensor from data loader
    label: ground-truth aim point
    pred: predited aim point
    global_step: iteration
    """
    import matplotlib.pyplot as plt
    import torchvision.transforms.functional as TF

    fig, ax = plt.subplots(1, 1)
    ax.imshow(TF.to_pil_image(img[0].cpu()))
    WH2 = np.array([img.size(-1), img.size(-2)]) / 2
    ax.add_artist(plt.Circle(WH2 * (label[0].cpu().detach().numpy() + 1), 2, ec="g", fill=False, lw=1.5))
    ax.add_artist(plt.Circle(WH2 * (pred[0].cpu().detach().numpy() + 1), 2, ec="r", fill=False, lw=1.5))
    logger.add_figure("viz", fig, global_step)
    del ax, fig

    if pred_peaks is None:
        return

    def hm_to_image(hm):
        clip = torch.clip(hm, 0.0, 1.0)
        return torch.stack((clip, clip, clip), dim=0)

    images = [
        hm_to_image(pred_peaks[0]),
    ]
    logger.add_images("image", torch.stack(images), global_step)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--log_dir", default="log")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--intense_augment", type=bool, default=True)
    parser.add_argument("--test_run", type=bool, default=False)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--profile", type=bool, default=False)
    parser.add_argument("--pretrained", type=str, default=None)

    args = parser.parse_args()

    if args.profile:
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_stack=True) as prof:
            train(args)

            print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=50))
            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=50))
            print(prof.key_averages(group_by_stack_n=5).table(sort_by="self_cuda_time_total", row_limit=5))
            prof.export_stacks("cuda_profiler_stacks.txt", "self_cuda_time_total")
            prof.export_stacks("cpu_profiler_stacks.txt", "self_cpu_time_total")
    else:
        train(args)
