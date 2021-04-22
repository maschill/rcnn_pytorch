import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict
from torch import device
from typing import Tuple


def _make_conv_block(filters_in, filters_out, kernel_size, padding=0):
    model = nn.Sequential(
        nn.Conv2d(filters_in, filters_out, kernel_size=3, padding=padding),
        nn.BatchNorm2d(filters_out),
        nn.ReLU(inplace=True),
    )
    return model


def calc_entropy(input_tensor):
    """
    https://github.com/pytorch/pytorch/issues/15829#issuecomment-725347711
    """
    return 11111
    lsm = nn.LogSoftmax(dim=1).to("cuda")
    log_probs = lsm(input_tensor)
    probs = torch.exp(log_probs)
    p_log_p = log_probs * probs
    entropy = -p_log_p.mean() * 1000
    # Multiplying by 1000 to bring it to 1-10 range.
    return entropy


class Bl_model(nn.Module):
    def __init__(
        self,
        num_classes: int = 10,
        input_dim: Tuple[int] = (128, 128, 3),
        steps: int = 1,
        unrolled: bool = False,
    ):
        """bottom-up/lateral convnet model with a simple conv2d-batchnorm-relu-maxpool 
        structure

        Args:
            num_classes (int, optional): number of classes to predict. Defaults to 10.
            input_dim (tuple, optional): input dim of images. Defaults to (128, 128, 3).
            steps (int, optional): how many recurrent steps. Defaults to 1.
            unrolled (bool, optional): deprecated. Defaults to False.
        """
        super().__init__()

        self.max_steps = steps
        self.eps = 0.0
        self.unrolled = unrolled
        unroll_name = ""
        if unrolled:
            unroll_name = "_unrolled"
        self.mname = f"bl_model_{steps}steps{unroll_name}"

        # init for input block
        block0_filters = 64
        self.input_block = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv0",
                        nn.Conv2d(
                            input_dim[-1], block0_filters, kernel_size=3, padding=1
                        ),
                    ),
                    ("bn0", nn.BatchNorm2d(block0_filters)),
                    ("relu0", nn.ReLU(inplace=True)),
                ]
            )
        )

        self.maxpool12 = nn.MaxPool2d(2)
        self.maxpool23 = nn.MaxPool2d(2)

        # init for first block, nn.ModuleDict so the model sees the layer in the list
        block1_filters = 128
        self.relu11 = nn.ReLU(inplace=True)
        self.relu12 = nn.ReLU(inplace=True)
        self.conv11 = nn.Conv2d(block0_filters, block1_filters, 3, padding=1)
        self.conv12 = nn.Conv2d(block1_filters, block1_filters, 3, padding=1)
        self.lateral1 = nn.Conv2d(block1_filters, block1_filters, 3, padding=1)
        self.bn11 = nn.ModuleDict(
            {f"bn11_{i}": nn.BatchNorm2d(block1_filters) for i in range(self.max_steps)}
        )
        self.bn12 = nn.ModuleDict(
            {f"bn12_{i}": nn.BatchNorm2d(block1_filters) for i in range(self.max_steps)}
        )

        # init for second block
        block2_filters = 256
        self.relu21 = nn.ReLU(inplace=True)
        self.relu22 = nn.ReLU(inplace=True)
        self.conv21 = nn.Conv2d(block1_filters, block2_filters, 3, padding=1)
        self.conv22 = nn.Conv2d(block2_filters, block2_filters, 3, padding=1)
        self.lateral2 = nn.Conv2d(block2_filters, block2_filters, 3, padding=1)
        self.bn21 = nn.ModuleDict(
            {f"bn21_{i}": nn.BatchNorm2d(block2_filters) for i in range(self.max_steps)}
        )
        self.bn22 = nn.ModuleDict(
            {f"bn22_{i}": nn.BatchNorm2d(block2_filters) for i in range(self.max_steps)}
        )

        # init for second block
        block3_filters = 512
        self.relu31 = nn.ReLU(inplace=True)
        self.relu32 = nn.ReLU(inplace=True)
        self.conv31 = nn.Conv2d(block2_filters, block3_filters, 3, padding=1)
        self.conv32 = nn.Conv2d(block3_filters, block3_filters, 3, padding=1)
        self.lateral3 = nn.Conv2d(block3_filters, block3_filters, 3, padding=1)
        self.bn31 = nn.ModuleDict(
            {f"bn31_{i}": nn.BatchNorm2d(block3_filters) for i in range(self.max_steps)}
        )
        self.bn32 = nn.ModuleDict(
            {f"bn32_{i}": nn.BatchNorm2d(block3_filters) for i in range(self.max_steps)}
        )

        # init for classifier
        self.avgpool = nn.ModuleDict(
            {
                f"avgpool_{i}": nn.AdaptiveAvgPool2d((1, 1))
                for i in range(self.max_steps)
            }
        )
        self.linear = nn.Linear(block3_filters, num_classes)

        if unrolled:
            print("unrolled is deprecated and should not be used")
            self.bn1_0 = self.bn1[0]
            self.bn1_1 = self.bn1[1]
            self.bn1_2 = self.bn1[2]
            self.bn2_0 = self.bn2[0]
            self.bn2_1 = self.bn2[1]
            self.bn2_2 = self.bn2[2]
            self.avgpool_0 = self.avgpool[0]
            self.avgpool_1 = self.avgpool[1]
            self.avgpool_2 = self.avgpool[2]

    def forward_looped(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        outputs = []

        x = self.input_block(x)

        x_in = self.conv11(x)
        x1 = self.bn11["bn11_0"](x_in)
        x1 = self.relu11(x1)
        x1 = self.conv12(x1)
        x1 = self.bn12["bn12_0"](x1)
        x1 = self.relu12(x1)

        xm2 = self.maxpool12(x1)

        x2_sum = self.conv21(xm2)
        x2 = self.bn21["bn21_0"](x2_sum)
        x2 = self.relu21(x2)
        x2 = self.conv22(x2)
        x2 = self.bn22["bn22_0"](x2)
        x2 = self.relu22(x2)

        xm3 = self.maxpool23(x2)

        x3_sum = self.conv31(xm3)
        x3 = self.bn31["bn31_0"](x3_sum)
        x3 = self.relu31(x3)
        x3 = self.conv32(x3)
        x3 = self.bn32["bn32_0"](x3)
        x3 = self.relu32(x3)

        out = self.avgpool["avgpool_0"](x3)
        out = out.view(out.size()[0], -1)
        cum_out = self.linear(out)

        outputs.append(cum_out)

        for t in range(1, self.max_steps):
            x1 = self.lateral1(x1)
            x1_sum = x_in + x1
            x1 = self.bn11[f"bn11_{t}"](x1_sum)
            x1 = self.relu11(x1)
            x1 = self.conv12(x1)
            x1 = self.bn12[f"bn12_{t}"](x1)
            x1 = self.relu12(x1)

            xm2 = self.maxpool12(x1)

            xl2 = self.lateral2(x2)
            x2_sum = self.conv21(xm2)
            x2_sum = x2_sum + xl2
            x2 = self.bn21[f"bn21_{t}"](x2_sum)
            x2 = self.relu21(x2)
            x2 = self.conv22(x2)
            x2 = self.bn22[f"bn22_{t}"](x2)
            x2 = self.relu22(x2)

            xm3 = self.maxpool23(x2)

            xl3 = self.lateral3(x3)
            x3_sum = self.conv31(xm3)
            x3_sum = x3_sum + xl3
            x3 = self.bn31[f"bn31_{t}"](x3_sum)
            x3 = self.relu31(x3)
            x3 = self.conv32(x3)
            x3 = self.bn32[f"bn32_{t}"](x3)
            x3 = self.relu32(x3)

            out = self.avgpool[f"avgpool_{t}"](x3)
            out = out.view(out.size()[0], -1)
            cum_out = self.linear(out)
            outputs.append(cum_out)

        return tuple(outputs)

    def forward_unrolled(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        """ deprecated smaller version of the model with 3 steps"""
        x = self.input_block(x)

        # first pass
        x_in = self.conv1(x)
        x1 = self.bn1_0(x_in)
        x1 = F.relu(x1)

        xm2 = self.maxpool(x1)
        x2_sum = self.conv2(xm2)
        x2 = self.bn2_0(x2_sum)
        x2 = F.relu(x2)

        out = self.avgpool_0(x2)
        out = out.view(out.size()[0], -1)
        out0 = self.linear(out)

        # second pass
        xl1 = self.lateral1(x1)
        x1_sum = x_in + xl1
        xl1 = self.bn1_1(x1_sum)
        xl1 = F.relu(xl1)

        xm2 = self.maxpool(xl1)
        x2_sum = self.conv2(xm2) + self.lateral2(x2)
        x2 = self.bn2_1(x2_sum)
        x2 = F.relu(x2)

        out = self.avgpool_1(x2)
        out = out.view(out.size()[0], -1)
        out1 = self.linear(out)

        # third pass
        xl1 = self.lateral1(xl1)
        x1_sum = x1_sum + x_in + xl1
        xl1 = self.bn1_2(x1_sum)
        xl1 = F.relu(xl1)

        xm2 = self.maxpool(xl1)
        x2_sum = x2_sum + self.conv2(xm2) + self.lateral2(x2)
        x2 = self.bn2_2(x2_sum)
        x2 = F.relu(x2)

        out = self.avgpool_2(x2)
        out = out.view(out.size()[0], -1)
        out2 = self.linear(out)

        return out0, out1, out2

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        if self.unrolled:
            return self.forward_unrolled(x)
        return self.forward_looped(x)

