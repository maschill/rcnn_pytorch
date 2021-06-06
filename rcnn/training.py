from contextlib import nullcontext
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
from rcnn import find_lr


def training(hparams: dict):
    """trains a model 

    Args:
        hparams (dict): a dictionary containing the hyperparameters
    """

    # currently, this is set manually to enable/disable pytorch profiler
    activate_profiler = False

    batch_size = hparams["batch_size"]
    num_epochs = hparams["num_epochs"]

    device = torch.device("cuda")
    # this model can be used instead
    # model = Bl_model(10, steps=hparams["steps"]).to(device)

    model = Bl_resnet(
        10,
        steps=hparams["steps"],
        threshold=torch.ones(8, device=device) * int(hparams["threshold"]),
        recurrence=hparams["recurrence"],
        residual=hparams["residual"],
    ).to(device)

    dataloaders = CIFAR10(batch_size, s=hparams["occlusion_size"])

    t0 = time.time()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=hparams["lr_start"])

    lr_scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        hparams["lr_start"],
        epochs=num_epochs,
        steps_per_epoch=(dataloaders.sizes["train"] // batch_size),
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
    writer.flush()
    # for tensorboard:
    hparams["recurrence"] = "".join([str(int(i)) for i in hparams["recurrence"]])

    # select context manager for profiling, or dummy context manager

    if activate_profiler:
        tb_profile_trace_handler = torch.profiler.tensorboard_trace_handler(
            str(writer_path.resolve())
        )
        cmgr = torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=2, warmup=2, active=3, repeat=1),
            on_trace_ready=tb_profile_trace_handler,
        )
    else:
        cmgr = nullcontext()

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

            tot = dataloaders.sizes[phase] // batch_size
            with cmgr as profiler:
                for batch, (inputs, labels) in tqdm(
                    enumerate(dataloaders[phase]), total=tot
                ):
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad(set_to_none=True)

                    with torch.set_grad_enabled(phase == "train"):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs[-1], 1)
                        loss = sum([criterion(o, labels) for o in outputs])
                        if phase == "train":
                            loss.backward()
                            optimizer.step()
                            lr_scheduler.step()
                        if activate_profiler and epoch == 0:
                            profiler.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataloaders.sizes[phase]
            epoch_acc = running_corrects.double() / dataloaders.sizes[phase]

            # tensorboard stuff
            writer.add_scalar("Loss/" + phase, epoch_loss, epoch)
            writer.add_scalar("Accuracy/" + phase, epoch_acc, epoch)

            if phase == "train":
                writer.add_scalar("LR/lr", current_lr, epoch)

                for name, param in model.named_parameters():
                    if "weight" in name and param.requires_grad:
                        if "bn" not in name:
                            if "lateral" not in name and "feedback" not in name:
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
        tot = dataloaders.sizes["val"] // batch_size * 2
        # 1024 total pixels > 1 % ~ 10
        for p, occ in enumerate([10 * i for i in range(0, 20)]):
            # init stuff
            run_val_loss = 0.0
            run_val_acc = 0.0
            dataloaders.update_val_loader(occ, mode=occ_mode)
            for batch, (inputs, labels) in tqdm(
                enumerate(dataloaders.val_loader), total=tot
            ):
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs[-1], 1)
                loss = sum([criterion(o, labels) for o in outputs])

                run_val_loss += loss.item() * inputs.size(0)
                run_val_acc += torch.sum(preds == labels.data)

            val_loss = run_val_loss / dataloaders.sizes["val"]
            val_acc = run_val_acc.double() / dataloaders.sizes["val"]
            print(f"Validation Loss: {val_loss:.4} Acc: {val_acc:.5}")

            metric_dict = {
                "hparam/loss": val_loss,
                "hparam/accuracy": val_acc,
            }

            writer.add_scalar(f"Accuracy/val_{occ_mode}", val_acc, p)

            writer.add_hparams(hparams, metric_dict, run_name="ht")
            writer.flush()
            if val_acc < 0.11:
                print(f"finished {occ_mode}----------------------")
                break

    writer.close()
    return 0

