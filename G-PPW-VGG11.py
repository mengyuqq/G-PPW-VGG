import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# class h_sigmoid(nn.Module):
#     def __init__(self, inplace=True):
#         super(h_sigmoid, self).__init__()
#         self.relu = nn.ReLU6(inplace=inplace)
#
#     def forward(self, x):
#         return self.relu(x + 3) / 6
#
#
# class h_swish(nn.Module):
#     def __init__(self, inplace=True):
#         super(h_swish, self).__init__()
#         self.sigmoid = h_sigmoid(inplace=inplace)
#
#     def forward(self, x):
#         return x * self.sigmoid(x)
#
#

# class depthwise_separable_conv(nn.Module):
#     def __init__(self, ch_in, ch_out):
#         super(depthwise_separable_conv, self).__init__()
#         self.ch_in = ch_in
#         self.ch_out = ch_out
#         self.depth_conv = nn.Conv2d(ch_in, ch_in, kernel_size=3, stride=1,padding=1, groups=ch_in)
#         self.point_conv = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1,padding=0, groups=1)
#
#     def forward(self, x):
#         x = self.depth_conv(x)
#         x = self.point_conv(x)
#         return x
class Partial_conv(nn.Module):
    def __init__(self, in_planes, n_div):
        super(Partial_conv, self).__init__()
        self.dim_conv3 = in_planes // n_div
        self.dim_untouched = in_planes - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)

    def forward(self, x):
        # for training/inference
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)
        return x


class CBA(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(CBA, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_planes,
                                 eps=0.001, # value found in tensorflow
                                 momentum=0.1, # default pytorch value
                                 affine=True)
        self.activate = nn.ReLU(inplace=True)

    def forward(self,x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activate(x)
        return x

class PM_Conv(nn.Module):
    def __init__(self, in_planes, n_div):
        super(PM_Conv, self).__init__()
        self.dim_conv = in_planes // n_div
        self.dim_untouched = in_planes - self.dim_conv - self.dim_conv
        self.partial_MixConv1 = nn.Conv2d(self.dim_conv, self.dim_conv, 3, 1, 1, bias=False)
        self.partial_MixConv2 = nn.Conv2d(self.dim_conv, self.dim_conv, 5, 1, 1, bias=False)

    def forward(self, x):
        # for training/inference
        x1, x2, x3 = torch.split(x, [self.dim_conv, self.dim_conv, self.dim_untouched], dim=1)
        x1 = self.partial_MixConv1(x1)
        x2 = self.partial_MixConv1(x2)
        x = torch.cat((x1, x2, x3), 1)
        return x

class ECA_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, k_size=3):
        super(ECA_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()

        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)

class vgg(nn.Module):
    def __init__(self, num_classes):
        super(vgg, self).__init__()
        self.num_classes = num_classes
        #self.act = h_swish()

        #卷积+激活+池化
        self.layer1 = nn.Sequential(
            CBA(3, 32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            CBA(32, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer3 = nn.Sequential(
            # CBA(64, 64, kernel_size=3, stride=1, padding=1),
            # CBA(64, 64, kernel_size=3, stride=1, padding=1),
            PM_Conv(64, 4),
            CBA(64, 64, kernel_size=1, stride=1, padding=0),
            PM_Conv(64, 4),
            CBA(64, 64, kernel_size=1, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # self.layer2 = Z_2()
        # self.layer3 = Z_3c()
        self.layer4 = nn.Sequential(
            # nn.Conv2d(160, 128,kernel_size=1, stride=1, padding=1),
            # CBA(64, 128, kernel_size=3, stride=1, padding=1),
            # CBA(128, 128, kernel_size=3, stride=1, padding=1),
            CBA(64, 128, kernel_size=1, stride=1, padding=0),
            # CBA(128, 128, kernel_size=3, stride=1, padding=1),
            Partial_conv(128, 4),
            CBA(128, 128, kernel_size=1, stride=1, padding=0),
            ECA_layer(128),
            Partial_conv(128, 4),
            CBA(128, 128, kernel_size=1, stride=1, padding=0),
            ECA_layer(128),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer5 = nn.Sequential(
            # CBA(128, 256, kernel_size=3, stride=1, padding=1),
            # CBA(256, 256, kernel_size=3, stride=1, padding=1),
            CBA(128, 256, kernel_size=1, stride=1, padding=0),
            Partial_conv(256, 4),
            CBA(256, 256, kernel_size=1, stride=1, padding=0),
            Partial_conv(256, 4),
            CBA(256, 256, kernel_size=1, stride=1, padding=0),

            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # self.layer6 = nn.Sequential(
        #     CBA(256, 256, kernel_size=1, stride=1, padding=0),
        #     CBA(256, 256, kernel_size=1, stride=1, padding=0),
        # )

        # self.layer6 = nn.Sequential(
        #     CBA(256, 512, kernel_size=3, stride=1, padding=1),
        #     CBA(512, 512, kernel_size=3, stride=1, padding=1),
        #     CBA(512, 512, kernel_size=3, stride=1, padding=1),
        #     nn.MaxPool2d(kernel_size=2, stride=2)
        # )
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        #全连接层
        self.avgpool = nn.Sequential(
            # nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=7, stride=1),
            # nn.Conv2d(256, 1024,kernel_size=3, stride=1),
            # nn.Conv2d(1024, 1024, kernel_size=3, stride=1),
            CBA(256, 256, kernel_size=1, stride=1, padding=0),
            CBA(256, 256, kernel_size=1, stride=1, padding=0),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Sequential(
            # CBA(256, 256, kernel_size=1, stride=1, padding=0),
            # CBA(256, 256, kernel_size=1, stride=1, padding=0),
            # nn.AdaptiveAvgPool2d((1, 1)),
            nn.Dropout(p=0.5),
            # nn.Linear(1024, 1024),
            # nn.BatchNorm1d(1024),
            # nn.ReLU(True),
            # nn.Dropout(p=0.5),
            nn.Linear(256, num_classes)
            # nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=7, stride=1),  # classifier.0
            # nn.ReLU(True),
            # nn.Dropout(p=0.5),
            # nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=1, stride=1),  # classifier.3
            # nn.ReLU(True),
            # nn.Dropout(p=0.5),
            # nn.Conv2d(in_channels=1024, out_channels=num_classes, kernel_size=1, stride=1)
        )
    def feature(self, input):
        x = self.layer1(input)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        # x = self.layer6(x)
        return x
    def forward(self,input):
        x = self.feature(input)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        # x = x.view(x.size(0), -1)
        x = self.fc(x)

        x = F.softmax(x, dim=1)  # 计算损失函数，输出概率最大的类别
        return x


# net = vgg(6)
# print(net)
# summary(net, (3, 244, 244))