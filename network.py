import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    """Implementation of the Arificial Neural Network using pytorch.
    """

    def __init__(self, args, frame_dim, mask):
        """Initialization of the network architecture

        Args:
            args (argparse): Parsed arguments
            frame_dim (tuple): Spatial Dimention of frame
            mask (numpy.ndarray): Region mask. Not in use.
        """
        super(Net, self).__init__()

        self.window_size = args.window_size
        self.dropout_prob = args.dropout
        self.ordinal = args.ordinal
        self.ordinal_resolution = args.ordinal_resolution
        self.use_harmonics = args.use_harmonics
        self.frame_dim = frame_dim  # (h, w)
        self.mask = mask
        self.use_mask = args.use_mask

        height_padding = self.calculate_same_padding(self.frame_dim[0], 3, 1)
        width_padding = self.calculate_same_padding(self.frame_dim[1], 3, 1)

        self.features3d = nn.Sequential(
            BasicConv3d(
                3,
                3,
                kernel_size=(1 + 2 * self.window_size, 3, 3),
                stride=(1, 1, 1),
                padding=(0, 2, 2),
            ),
            BasicConv3d(
                3, 8, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 2, 2),
            ),
        )

        self.features2d = nn.Sequential(
            BasicConv2d(8, 16, kernel_size=3, stride=1),
            BasicConv2d(16, 32, kernel_size=3, stride=1),
            BasicConv2d(32, 16, kernel_size=1, stride=1),
        )

        self.spp = SpatialPyramidPooling(5)

        self.dropout = nn.Dropout(p=self.dropout_prob)
        self.activation1 = SiLU()
        self.activation2 = SiLU()

        if self.use_harmonics:
            self.fc1 = nn.Linear(880 + 25, 256)
        else:
            self.fc1 = nn.Linear(880, 256)
        self.fc2 = nn.Linear(256, 50)
        self.out = nn.Linear(50, 1)

        if self.use_harmonics:
            self.fc3 = nn.Linear(880 + 25, 256)
        else:
            self.fc3 = nn.Linear(880, 256)
        self.fc4 = nn.Linear(256, 50)
        self.out_ordinal = nn.Linear(50, self.ordinal_resolution)

    def forward(self, x, series=None):
        if self.use_mask:
            mask = torch.from_numpy(self.mask.astype(np.float32))
            mask = mask.to("cuda:1")

        x = self.features3d(x)
        if self.use_mask:
            x = x * mask
        x = x.view(x.size(0), x.size(1), x.size(3), x.size(4))  # 3d to 2d
        x = self.features2d(x)

        if self.use_mask:
            x = x * mask

        x = self.spp(x)

        if self.use_harmonics:
            x = torch.cat((x, series), 1)

        # ordinal
        if self.ordinal:
            x = self.dropout(self.activation1(self.fc3(x)))
            x = self.dropout(self.activation2(self.fc4(x)))
            second_to_last_layer = x
            # do not use this if BCEWithLogitsLoss is used as loss:
            # x = torch.sigmoid(self.out_ordinal(x))
            x = self.out_ordinal(x)
        else:
            x = self.dropout(self.activation1(self.fc1(x)))
            x = self.dropout(self.activation2(self.fc2(x)))
            second_to_last_layer = x
            x = torch.sigmoid(self.out(x))

        return x, second_to_last_layer

    def calculate_same_padding(self, W, F, S):
        # P = ((S-1)*W-S+F)/2, with F = filter size, S = stride
        padding = int(np.ceil(((S - 1) * W - S + F) / 2))
        return padding


class SpatialPyramidPooling(nn.Module):
    def __init__(self, level):
        """Implementation of the Spatial Pyramid Pooling layer, He et al. [2015].

        Args:
            level (int): Number of levels in the SPP layer. Usially 3-5.
        """
        super(SpatialPyramidPooling, self).__init__()
        self.n = level

    def forward(self, x):
        n = self.n
        shape = x.size()
        # out = torch.Tensor()  # fiks dette. try to get everything solved without useing lists.
        kernel, stride = self.calculate_kenel_and_stride(shape, n)
        out = torch.nn.functional.max_pool2d(x, kernel, stride).view(x.size(0), -1)
        n -= 1
        while n > 0:
            kernel, stride = self.calculate_kenel_and_stride(shape, n)

            # to fix overperfoming, average pooling can be tested instead of
            # maxpool.
            tmp = torch.nn.functional.max_pool2d(x, kernel, stride)
            out = torch.cat((out, tmp.view(x.size(0), -1),), 1,)
            n -= 1
        return out

    def calculate_kenel_and_stride(self, shape, n):
        ah = shape[2]
        aw = shape[3]

        kernel_h = int(np.ceil(ah / n))
        kernel_w = int(np.ceil(aw / n))

        stride_h = int(np.floor(ah / n))
        stride_w = int(np.floor(aw / n))

        return (kernel_h, kernel_w), (stride_h, stride_w)


class BasicConv3d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(BasicConv3d, self).__init__()
        self.conv = nn.Conv3d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.bn = nn.BatchNorm3d(out_planes)
        self.activation = SiLU()
        # self.activation = nn.LeakyReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.bn = nn.BatchNorm2d(out_planes)
        self.activation = SiLU()
        # self.activation = nn.LeakyReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


def silu(x, beta):
    return x * torch.sigmoid(beta * x)


class SiLU(nn.Module):
    def __init__(self, beta=None):
        """Swish activation function with beta = 1
        """
        super(SiLU, self).__init__()
        if beta == None:
            self.beta = nn.Parameter(torch.tensor(1.0))
        else:
            self.beta = nn.Parameter(torch.tensor(beta))
        self.beta.requiresGrad = False

    def forward(self, x):
        return silu(x, self.beta)
