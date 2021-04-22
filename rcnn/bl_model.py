import torch.nn as nn
import torch.nn.functional as F

from torch import device


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
        self, num_classes=10, input_dim=(128, 128, 3), steps=1, unrolled=False
    ):
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
            nn.Conv2d(input_dim[-1], block0_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(block0_filters),
            nn.ReLU(inplace=True),
        )

        self.maxpool = nn.MaxPool2d(2)

        # init for first block, nn.ModuleList so the model sees the layer in the list
        block1_filters = 128
        self.conv1 = nn.Conv2d(block0_filters, block1_filters, 3, padding=1)
        self.conv12 = nn.Conv2d(block1_filters, block1_filters, 3, padding=1)
        self.lateral1 = nn.Conv2d(block1_filters, block1_filters, 3, padding=1)
        self.bn1 = nn.ModuleList(
            [nn.BatchNorm2d(block1_filters) for _ in range(self.max_steps)]
        )
        self.bn12 = nn.ModuleList(
            [nn.BatchNorm2d(block1_filters) for _ in range(self.max_steps)]
        )

        # init for second block
        block2_filters = 256
        self.conv2 = nn.Conv2d(block1_filters, block2_filters, 3, padding=1)
        self.conv22 = nn.Conv2d(block2_filters, block2_filters, 3, padding=1)
        self.lateral2 = nn.Conv2d(block2_filters, block2_filters, 3, padding=1)
        self.bn2 = nn.ModuleList(
            [nn.BatchNorm2d(block2_filters) for _ in range(self.max_steps)]
        )
        self.bn22 = nn.ModuleList(
            [nn.BatchNorm2d(block2_filters) for _ in range(self.max_steps)]
        )

        # init for second block
        block3_filters = 512
        self.conv3 = nn.Conv2d(block2_filters, block3_filters, 3, padding=1)
        self.conv32 = nn.Conv2d(block3_filters, block3_filters, 3, padding=1)
        self.lateral3 = nn.Conv2d(block3_filters, block3_filters, 3, padding=1)
        self.bn3 = nn.ModuleList(
            [nn.BatchNorm2d(block3_filters) for _ in range(self.max_steps)]
        )
        self.bn32 = nn.ModuleList(
            [nn.BatchNorm2d(block3_filters) for _ in range(self.max_steps)]
        )

        # init for classifier
        self.avgpool = nn.ModuleList(
            [nn.AdaptiveAvgPool2d((1, 1)) for _ in range(self.max_steps)]
        )
        self.linear = nn.Linear(block2_filters, num_classes)

        if unrolled:
            self.bn1_0 = self.bn1[0]
            self.bn1_1 = self.bn1[1]
            self.bn1_2 = self.bn1[2]
            self.bn2_0 = self.bn2[0]
            self.bn2_1 = self.bn2[1]
            self.bn2_2 = self.bn2[2]
            self.avgpool_0 = self.avgpool[0]
            self.avgpool_1 = self.avgpool[1]
            self.avgpool_2 = self.avgpool[2]

    def forward_looped(self, x):
        outputs = []

        x = self.input_block(x)

        x_in = self.conv1(x)
        x1 = self.bn1[0](x_in)
        x1 = F.relu(x1)
        x1 = self.conv12(x1)
        x1 = self.bn12[0](x1)
        x1 = F.relu(x1)

        xm2 = self.maxpool(x1)

        x2_sum = self.conv2(xm2)
        x2 = self.bn2[0](x2_sum)
        x2 = F.relu(x2)
        x2 = self.conv22(x2)
        x2 = self.bn22[0](x2)
        x2 = F.relu(x2)

        xm3 = self.maxpool(x2)

        x3_sum = self.conv2(xm3)
        x3 = self.bn3[0](x3_sum)
        x3 = F.relu(x3)
        x3 = self.conv32(x3)
        x3 = self.bn32[0](x3)
        x3 = F.relu(x3)

        out = self.avgpool[0](x3)
        out = out.view(out.size()[0], -1)
        cum_out = self.linear(out)

        outputs.append(cum_out)

        for t in range(1, self.max_steps):
            x1 = self.lateral1(x1)
            x1_sum = x_in + x1
            x1 = self.bn1[t](x1_sum)
            x1 = F.relu(x1)
            x1 = self.conv12(x1)
            x1 = self.bn12[t](x1)
            x1 = F.relu(x1)

            xm2 = self.maxpool(x1)

            xl2 = self.lateral2(x2)
            x2_sum = self.conv2(xm2)
            x2_sum = x2_sum + xl2
            x2 = self.bn2[t](x2_sum)
            x2 = F.relu(x2)
            x2 = self.conv22(x2)
            x2 = self.bn22[t](x2)
            x2 = F.relu(x2)

            xm3 = self.maxpool(x2)

            xl3 = self.lateral3(x3)
            x3_sum = self.conv3(xm3)
            x3_sum = x3_sum + xl3
            x3 = self.bn3[t](x3_sum)
            x3 = F.relu(x3)
            x3 = self.conv32(x3)
            x3 = self.bn32[t](x3)
            x3 = F.relu(x3)

            out = self.avgpool[t](x3)

            out = self.avgpool[t](x2)
            out = out.view(out.size()[0], -1)
            cum_out = self.linear(out)
            outputs.append(cum_out)

        return tuple(outputs)

    def forward_unrolled(self, x):
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

    def forward(self, x):
        if self.unrolled:
            return self.forward_unrolled(x)
        return self.forward_looped(x)

