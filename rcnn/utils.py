import torch

import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from typing import Generator
from typing import Tuple

import matplotlib

from rcnn import DataContainer

matplotlib.use("Agg")


def winit(w_id: int):
    """helper function that creates a np.random seed that is different for 
        each worker used in a pytorch DataLoader. For more infos check
        https://tanelp.github.io/posts/a-bug-that-plagues-thousands-of-open-source-ml-projects/

    Args:
        w_id (int): id of the created worker
    """
    np.random.seed(np.random.get_state()[1][0] + w_id)


def find_lr(model: nn.Module, dataloaders: DataContainer) -> Tuple[float, float]:
    """Tests a range of 10e-8 to 2 to find a good learning rate 

    Args:
        model (nn.Module): the model the lr is used for
        dataloaders (DataContainer): the DataContainer that supplies the data

    Returns:
        Tuple[float, float]: two candidates for a learning rate.
                            The one with the steepest gradient and 
                            the one with the smallest loss
    """
    lr_start = 10e-8
    lr_end = 2

    model = model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=lr_start)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, 1, 1.1)
    criterion = nn.CrossEntropyLoss()
    losses = []
    lrs = []

    for inputs, labels in dataloaders["train"]:
        lrs.append(optimizer.param_groups[0]["lr"])

        inputs = inputs.cuda()
        labels = labels.cuda()

        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            outputs = model(inputs)
            loss = sum([criterion(o, labels) for o in outputs])
            loss.backward()
            optimizer.step()
            losses.append(loss)

        lr_scheduler.step()
        print(lrs[-1], end="\t\t\r")
        if (losses[-1] > 5 * model.max_steps) or (lrs[-1] > lr_end):
            break

    losses = torch.stack(losses).cpu()
    w = 5
    w2 = w // 2
    ls = sum(losses[0:w]) / w
    lsmooth = [ls]

    for i in range(w2 + 1, len(losses) - w2):
        ls = (ls * w - losses[i - w2 - 1] + losses[i + w2]) / w
        lsmooth.append(ls)

    lsmooth = torch.FloatTensor(lsmooth)
    lrs = torch.FloatTensor(lrs)[w2:-w2]

    grads = (lsmooth[1:] - lsmooth[:-1]) / (lrs[1:].log() - lrs[:-1].log())
    gmax, lmin = (
        float(lrs[grads.argmin()].item()),
        float(lrs[lsmooth.argmin()].item() / 10),
    )

    plt.figure()
    plt.plot(
        lrs.detach().cpu().numpy(), losses[w2:-w2].detach().cpu().numpy(), label="raw"
    )
    plt.plot(lrs.detach().cpu().numpy(), lsmooth, label=f"moving average w={w}")
    plt.scatter(
        [lmin * 10, gmax],
        [lsmooth[lsmooth.argmin()].item(), lsmooth[grads.argmin()].item()],
    )
    plt.xscale("log")
    plt.legend()
    plt.savefig(f"lrfinder_{model.mname}.png")
    print(f"min_loss at LR {lmin}")
    print(f"steepest grad at {gmax}")

    return gmax, lmin


def calc_entropy(input_tensor: torch.Tensor) -> torch.Tensor:
    """calculates the entropy of a Tensor, code from
    https://github.com/pytorch/pytorch/issues/15829#issuecomment-725347711

    Args:
        input_tensor (torch.Tensor): Tensor the entropy is calculated of

    Returns:
        torch.Tensor: The entropy of the input_tensor
    """
    lsm = nn.LogSoftmax(dim=1).to("cuda")
    log_probs = lsm(input_tensor)
    probs = torch.exp(log_probs)
    p_log_p = log_probs * probs
    # Multiplying by 1000 to bring it to a better range.
    entropy = -p_log_p.mean() * 1000
    return entropy


def rand_arr(max_occ: int = 32) -> Generator[int, None, None]:
    """Generator that bulk creates random numbers and returns them one by one
        for the Occlusion transformation, to redus np.random calls 

    Args:
        max_occ (int, optional): maximum value. Defaults to 32.

    Yields:
        Generator[int, None, None]: a generator that yields a random int
    """
    while True:
        randr = np.random.randint(0, max_occ, 500000)
        for i in randr:
            yield i


class Occlusion:
    """Tranformation to add occlusion to the preprocessing"""

    def __init__(self, mode: str = "cutout", size: int = 4, val: int = 150):
        """
        Args:
            mode (str, optional): occlusion mode, can be "cutout" or "noise.
                                Defaults to "cutout".
            size (int, optional): the amount of occlusion used. Number 
                                of occluded pixels in noise or side length
                                in pixels of occluded square in "cutout" .
                                Defaults to 4.
            val (int, optional): [description].Color val of occlusion. Defaults to 150.
        """
        self.mode = mode
        self.size = size
        self.val = val
        if mode == "cutout":
            self.rand = iter(rand_arr(max_occ=max(1, 32 - size)))
        else:
            self.rand = iter(rand_arr(max_occ=max(1, 30)))

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """adds occlusion to input image

        Args:
            img (torch.Tensor): input image

        Returns:
            torch.Tensor: input image with occlusion
        """
        if self.mode == "cutout":
            ix, iy = next(self.rand), next(self.rand)
            img[:, ix : ix + self.size, iy : iy + self.size] = self.val
            return img
        elif self.mode == "noise":
            for _ in range(self.size):
                img[:, next(self.rand), next(self.rand)] = self.val
            return img
        return img

