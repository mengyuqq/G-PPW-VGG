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

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activate(x)
        return x

# class DBA(nn.Module):
#     def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
#         super(DBA, self).__init__()
#         self.conv = depthwise_separable_conv(in_planes, out_planes)
#         # self.bn = nn.BatchNorm2d(out_planes,
#         #                          eps=0.001, # value found in tensorflow
#         #                          momentum=0.1, # default pytorch value
#         #                          affine=True)
#         self.activate = nn.ReLU(inplace=True)
#
#     def forward(self,x):
#         x = self.conv(x)
#         # x = self.bn(x)
#         x = self.activate(x)
#         return x
#
# class Z_2(nn.Module):
#
#     def __init__(self):
#         super(Z_2, self).__init__()
#         self.maxpool = nn.MaxPool2d(3, stride=2)
#         self.conv = CBA(64, 96, kernel_size=3, stride=2)
#
#     def forward(self, x):
#         x0 = self.maxpool(x)
#         x1 = self.conv(x)
#         out = torch.cat((x0, x1), 1)
#         return out
# class Z_3a(nn.Module):
#
#     def __init__(self):
#         super(Z_3a, self).__init__()
#
#         self.branch0 = nn.Sequential(
#             CBA(160, 64, kernel_size=1, stride=1),
#             CBA(64, 96, kernel_size=3, stride=1)
#         )
#
#         self.branch1 = nn.Sequential(
#             CBA(160, 64, kernel_size=1, stride=1),
#             CBA(64, 64, kernel_size=(1,7), stride=1, padding=(0,3)),
#             CBA(64, 64, kernel_size=(7,1), stride=1, padding=(3,0)),
#             CBA(64, 96, kernel_size=(3,3), stride=1)
#         )
#
#     def forward(self, x):
#         x0 = self.branch0(x)
#         x1 = self.branch1(x)
#         out = torch.cat((x0, x1), 1)
#         return out
#
#
# class Z_3b(nn.Module):
#
#     def __init__(self):
#         super(Z_3b, self).__init__()
#         self.conv = CBA(192, 192, kernel_size=3, stride=2)
#         self.maxpool = nn.MaxPool2d(3, stride=2)
#
#     def forward(self, x):
#         x0 = self.conv(x)
#         x1 = self.maxpool(x)
#         out = torch.cat((x0, x1), 1)
#         return out

# class Z_3c(nn.Module):
#
#     def __init__(self):
#         super(Z_3c, self).__init__()
#         self.branch0 = nn.Sequential(
#             CBA(160, 64, kernel_size=1, stride=1),
#             nn.MaxPool2d(3, stride=2)
#         )
#
#         self.branch1 = nn.Sequential(
#             CBA(160, 64, kernel_size=1, stride=1),
#             CBA(64, 64, kernel_size=(1, 7), stride=1, padding=(0, 3)),
#             CBA(64, 64, kernel_size=(7, 1), stride=1, padding=(3, 0)),
#             CBA(64, 96, kernel_size=(3, 3), stride=2)
#         )
#
#     def forward(self, x):
#         x0 = self.branch0(x)
#         x1 = self.branch1(x)
#         out = torch.cat((x0, x1), 1)
#         return out

class vgg(nn.Module):
    def __init__(self, num_classes):
        super(vgg, self).__init__()
        self.num_classes = num_classes
        #self.act = h_swish()

        #卷积+激活+池化
        self.layer1 = nn.Sequential(
            CBA(3, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            CBA(64, 128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer3 = nn.Sequential(
            CBA(128, 256, kernel_size=3, stride=1, padding=1),
            CBA(256, 256, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # self.layer2 = Z_2()
        # self.layer3 = Z_3c()
        self.layer4 = nn.Sequential(
            # nn.Conv2d(160, 128,kernel_size=1, stride=1, padding=1),
            CBA(256, 512, kernel_size=3, stride=1, padding=1),
            CBA(512, 512, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer5 = nn.Sequential(
            CBA(512, 512, kernel_size=3, stride=1, padding=1),
            CBA(512, 512, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # self.layer6 = nn.Sequential(
        #     CBA(256, 512, kernel_size=3, stride=1, padding=1),
        #     CBA(512, 512, kernel_size=3, stride=1, padding=1),
        #     CBA(512, 512, kernel_size=3, stride=1, padding=1),
        #     nn.MaxPool2d(kernel_size=2, stride=2)
        # )
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        #全连接层
        self.fc = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)
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
        x = x.view(x.size(0), -1)
        # x = torch.flatten(x, start_dim=1)
        x = self.fc(x)

        # x = F.log_softmax(x, dim=1)  # 计算损失函数，输出概率最大的类别
        return x

# net = vgg(6)
# print(net)
# summary(net, (3, 244, 244))