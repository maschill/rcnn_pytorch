import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List

from rcnn import calc_entropy


class Bl_resnet(nn.Module):
    def __init__(
        self,
        num_classes: int,
        steps: int = 1,
        threshold: int = 100,
        recurrence: list = [True, False, True],
        residual: bool = True,
        pooling: nn.Module = nn.MaxPool2d,
    ):
        """ bottom-up/lateral convnet Model based on resnet18 for 
            image classification

        Args:
            num_classes (int): number of classes for image classification 
            steps (int, optional):  recurrent steps. 1 means feedforward. 
                                    needs to be >=1. Defaults to 1.
            threshold (int, optional): if recurrent, the threshold that 
                        entropy needs to be below of for early output. 
                        Defaults to 100.
            recurrence (list, optional): needs to be a list of booleans 
                        of length 3. boolean flags wether a block has a 
                        lateral (recurrent) connection. 
                        Defaults to [True,False,True].
            residual (bool, optional): wether or not residual connections 
                        are enabled. Defaults to True.
            pooling (nn.Module, optional): what pooling layer to use. 
                        Defaults to nn.MaxPool2d.
        """
        super().__init__()

        self.max_steps = steps
        self.threshold = threshold
        self.mname = (
            f"bl{''.join([str(int(r)) for r in recurrence])}_resnet_{steps}steps"
        )
        self.residual = residual
        self.recurrence = recurrence

        # init for input block
        block0_filters = 64
        self.input_block = nn.Sequential(
            nn.Conv2d(3, block0_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(block0_filters),
            nn.ReLU(),
        )

        block1_filters = 128
        block2_filters = block1_filters * 2
        block3_filters = block2_filters * 2

        # init for first block
        self.conv1 = nn.Conv2d(block0_filters, block1_filters, 3, padding=1)
        self.bn1 = nn.ModuleList(
            [nn.BatchNorm2d(block1_filters) for _ in range(self.max_steps)]
        )
        self.conv12 = nn.Conv2d(block1_filters, block1_filters, 3, padding=1)
        self.bn12 = nn.ModuleList(
            [nn.BatchNorm2d(block1_filters) for _ in range(self.max_steps)]
        )
        self.maxpool1 = pooling(2)

        if self.recurrence[0]:
            self.lateral1 = nn.Conv2d(block1_filters, block1_filters, 3, padding=1)

        # init for second block
        self.conv2 = nn.Conv2d(block1_filters, block2_filters, 3, padding=1)
        self.bn2 = nn.ModuleList(
            [nn.BatchNorm2d(block2_filters) for _ in range(self.max_steps)]
        )
        self.conv22 = nn.Conv2d(block2_filters, block2_filters, 3, padding=1)
        self.bn22 = nn.ModuleList(
            [nn.BatchNorm2d(block2_filters) for _ in range(self.max_steps)]
        )
        self.maxpool2 = pooling(2)

        if self.recurrence[1]:
            self.lateral2 = nn.Conv2d(block2_filters, block2_filters, 3, padding=1)

        # init for third block
        self.conv3 = nn.Conv2d(block2_filters, block3_filters, 3, padding=1)
        self.bn3 = nn.ModuleList(
            [nn.BatchNorm2d(block3_filters) for _ in range(self.max_steps)]
        )
        self.conv32 = nn.Conv2d(block3_filters, block3_filters, 3, padding=1)
        self.bn32 = nn.ModuleList(
            [nn.BatchNorm2d(block3_filters) for _ in range(self.max_steps)]
        )

        if self.recurrence[2]:
            self.lateral3 = nn.Conv2d(block3_filters, block3_filters, 3, padding=1)

        # init res con
        if self.residual:
            self.res1bn = nn.ModuleList(
                [nn.BatchNorm2d(block1_filters) for _ in range(self.max_steps)]
            )
            self.res1 = nn.Conv2d(block0_filters, block1_filters, 3, padding=1)
            self.res2bn = nn.ModuleList(
                [nn.BatchNorm2d(block2_filters) for _ in range(self.max_steps)]
            )
            self.res2 = nn.Conv2d(block1_filters, block2_filters, 3, padding=1)
            self.res3bn = nn.ModuleList(
                [nn.BatchNorm2d(block3_filters) for _ in range(self.max_steps)]
            )
            self.res3 = nn.Conv2d(block2_filters, block3_filters, 3, padding=1)

        # init for output block
        self.avgpool = nn.ModuleList(
            [nn.AdaptiveAvgPool2d((1, 1)) for _ in range(self.max_steps)]
        )
        self.linear = nn.Linear(block3_filters, num_classes)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """forward step of the model

        Args:
            x (torch.Tensor): input of shape BxCxHxW

        Returns:
            list: list of vectors representing the classification at each step
        """
        outputs = []
        x = self.input_block(x)
        x_in = self.conv1(x)
        if self.residual:
            res1 = self.res1(x)
        l1 = 0
        l2 = 0
        l3 = 0

        for t in range(0, self.max_steps):
            x1sum = x_in
            if self.recurrence[0]:
                x1sum = x1sum + l1
            x1 = self.bn1[t](x1sum)
            x1 = F.celu(x1, alpha=0.01)
            x1 = self.conv12(x1)
            x1 = self.bn12[t](x1)
            x1 = F.celu(x1, alpha=0.01)
            if self.residual:
                res1 = self.res1bn[t](res1)
                x1 = x1 + res1

            if self.recurrence[0] and t < self.max_steps:  # don't calc in last step
                l1 = self.lateral1(x1)

            x12 = self.maxpool1(x1)

            x2sum = self.conv2(x12)
            if self.recurrence[1]:
                x2sum = x2sum + l2
            x2 = self.bn2[t](x2sum)
            x2 = F.celu(x2, alpha=0.01)
            x2 = self.conv22(x2)
            x2 = self.bn22[t](x2)
            x2 = F.celu(x2, alpha=0.01)
            if self.residual:
                res2 = self.res2(x12)
                res2 = self.res2bn[t](res2)
                x2 = x2 + res2
            if self.recurrence[1] and t < self.max_steps:  # don't calc in last step
                l2 = self.lateral2(x2)

            x23 = self.maxpool2(x2)

            x3sum = self.conv3(x23)
            if self.recurrence[2]:
                x3sum = x3sum + l3
            x3 = self.bn3[0](x3sum)
            x3 = F.celu(x3, alpha=0.01)
            x3 = self.conv32(x3)
            x3 = self.bn32[t](x3)
            x3 = F.celu(x3, alpha=0.01)
            if self.residual:
                res3 = self.res3(x23)
                res3 = self.res3bn[t](res3)
                x3 = x3 + res3
            if self.recurrence[2] and t < self.max_steps:  # don't calc in last step
                l3 = self.lateral3(x3)

            out = self.avgpool[t](x3)
            out = out.view(out.size()[0], -1)
            out = self.linear(out)
            outputs.append(out)

            ent = calc_entropy(out)

            if ent < self.threshold[0]:
                break

        return outputs
