from datetime import datetime
import os
import time
import torch
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from rcnn import Bl_resnet
from rcnn import Bl_model
from rcnn import CIFAR10


def training(hparams: dict):
    """trains a model 

    Args:
        hparams (dict): a dictionary containing the hyperparameters
    """

    batch_size = hparams["batch_size"]
    num_epochs = hparams["num_epochs"]

    device = torch.device("cuda")

    # model = Bl_model(10, steps=hparams["steps"]).to(device)

    model = Bl_resnet(
        10,
        steps=hparams["steps"],
        threshold=torch.FloatTensor(hparams["threshold"]).to(device),
        recurrence=hparams["recurrence"],
        residual=hparams["residual"],
    ).to(device)

    dl = CIFAR10(batch_size, s=hparams["occlusion_size"])
    dataloaders = dl.dl_dict()
    dataset_sizes = dl.sizes

    t0 = time.time()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=hparams["lr_start"])

    lr_scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        hparams["lr_start"],
        epochs=num_epochs,
        steps_per_epoch=(dataset_sizes["train"] // batch_size),
    )

    starttime = f"started_{datetime.now().strftime('%d-%m-%Y %H:%M:%S')}"
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"{num_params/1000000:.4}M")
    output_subdir = (
        Path("cifar10")
        / f"{model.mname}_{num_params/1000000:.2}M"
        / f"{num_epochs}_ep"
        / f"BS_{batch_size}_occlusion_{hparams['occlusion_size']}"
        / f"lr_{hparams['lr_start']}_threshold-{hparams['threshold']}"
        / starttime
    )

    writer_path = Path("Runs") / output_subdir
    writer = SummaryWriter(str(writer_path.resolve()))
    writer.add_graph(model, torch.zeros(1, 3, 32, 32).cuda(), verbose=False)

    # for tensorboard:
    hparams["recurrence"] = "".join([str(int(i)) for i in hparams["recurrence"]])

    for epoch in range(num_epochs):
        current_lr = optimizer.param_groups[0]["lr"]
        print("-" * 10)
        print(f"Epoch {epoch+1}/{num_epochs}, LR: {current_lr}")
        print("-" * 10)

        for phase in ["train", "test"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            tot = dataset_sizes[phase] // batch_size
            for batch, (inputs, labels) in tqdm(
                enumerate(dataloaders[phase]), total=tot
            ):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs[-1], 1)
                    loss = sum([criterion(o, labels) for o in outputs])
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                        lr_scheduler.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            # tensorboard stuff
            writer.add_scalar("Loss/" + phase, epoch_loss, epoch)
            writer.add_scalar("Accuracy/" + phase, epoch_acc, epoch)

            if phase == "train":
                writer.add_scalar("LR/lr", current_lr, epoch)

                for name, param in model.named_parameters():
                    if "weight" in name and param.requires_grad:
                        if "bn" not in name:
                            if "lateral" not in name or hparams["steps"] > 1:
                                writer.add_histogram(name, param, epoch)
                                writer.add_histogram(name + ".grad", param.grad, epoch)

            print(f"{phase} Loss: {epoch_loss:.4} Acc: {epoch_acc:.5}")

        writer.flush()

    t1 = time.time()
    print(f"Total training time {t1-t0:.4} seconds")

    print("saving model")

    ckpt_path = Path("ckpt") / output_subdir

    os.makedirs(ckpt_path)
    torch.save(model.state_dict(), ckpt_path / "model.pt")

    print("... validating ...")
    for occ_mode in ["cutout", "noise"]:
        model.eval()
        tot = dataset_sizes["val"] // batch_size * 2
        # 1024 total pixels > 1 % ~ 10
        for p, occ in enumerate([10 * i for i in range(0, 20)]):
            # init stuff
            run_val_loss = 0.0
            run_val_acc = 0.0
            dl.update_val_loader(occ, mode=occ_mode)
            for batch, (inputs, labels) in tqdm(enumerate(dl.val_loader), total=tot):
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs[-1], 1)
                loss = sum([criterion(o, labels) for o in outputs])

                run_val_loss += loss.item() * inputs.size(0)
                run_val_acc += torch.sum(preds == labels.data)

            val_loss = run_val_loss / dataset_sizes["val"]
            val_acc = run_val_acc.double() / dataset_sizes["val"]
            print(f"Validation Loss: {val_loss:.4} Acc: {val_acc:.5}")

            metric_dict = {
                "hparam/loss": val_loss,
                "hparam/accuracy": val_acc,
            }

            writer.add_scalar(f"Accuracy/val_{occ_mode}", val_acc, p)

            writer.add_hparams(hparams, metric_dict, run_name="ht")
            writer.flush()
            if val_acc < 0.1:
                print(f"finished {occ_mode}----------------------")
                break

    writer.close()
    return 0

