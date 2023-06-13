"""
3D Squeeze and Excitation Modules
*****************************
3D Extensions of the following 2D squeeze and excitation blocks:

    1. `Channel Squeeze and Excitation <https://arxiv.org/abs/1709.01507>`_
    2. `Spatial Squeeze and Excitation <https://arxiv.org/abs/1803.02579>`_
    3. `Channel and Spatial Squeeze and Excitation <https://arxiv.org/abs/1803.02579>`_

New Project & Excite block, designed specifically for 3D inputs
    'quote'

    Coded by -- Anne-Marie Rickmann (https://github.com/arickm)
"""

from enum import Enum
import torch
from torch import nn as nn
from torch.nn import functional as F


class ChannelSELayer3D(nn.Module):
    """
    3D extension of Squeeze-and-Excitation (SE) block described in:
        *Hu et al., Squeeze-and-Excitation Networks, arXiv:1709.01507*
        *Zhu et al., AnatomyNet, arXiv:arXiv:1808.05238*
    """

    def __init__(self, num_channels, reduction_ratio=2):
        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ChannelSELayer3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, D, H, W)
        :return: output tensor
        """
        batch_size, num_channels, D, H, W = input_tensor.size()
        # Average along each channel
        squeeze_tensor = self.avg_pool(input_tensor)

        # channel excitation
        fc_out_1 = self.relu(self.fc1(squeeze_tensor.view(batch_size, num_channels)))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        output_tensor = torch.mul(input_tensor, fc_out_2.view(batch_size, num_channels, 1, 1, 1))

        return output_tensor


class SpatialSELayer3D(nn.Module):
    """
    3D extension of SE block -- squeezing spatially and exciting channel-wise described in:
        *Roy et al., Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks, MICCAI 2018*
    """

    def __init__(self, num_channels):
        """
        :param num_channels: No of input channels

        """
        super(SpatialSELayer3D, self).__init__()
        self.conv = nn.Conv3d(num_channels, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor, weights=None):
        """
        :param weights: weights for few shot learning
        :param input_tensor: X, shape = (batch_size, num_channels, D, H, W)
        :return: output_tensor
        """
        # channel squeeze
        batch_size, channel, D, H, W = input_tensor.size()

        if weights:
            weights = weights.view(1, channel, 1, 1)
            out = F.conv2d(input_tensor, weights)
        else:
            out = self.conv(input_tensor)

        squeeze_tensor = self.sigmoid(out)

        # spatial excitation
        output_tensor = torch.mul(input_tensor, squeeze_tensor.view(batch_size, 1, D, H, W))

        return output_tensor


class ChannelSpatialSELayer3D(nn.Module):
    """
       3D extension of concurrent spatial and channel squeeze & excitation:
           *Roy et al., Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks, arXiv:1803.02579*
       """

    def __init__(self, num_channels, reduction_ratio=2):
        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ChannelSpatialSELayer3D, self).__init__()
        self.cSE = ChannelSELayer3D(num_channels, reduction_ratio)
        self.sSE = SpatialSELayer3D(num_channels)

    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, D, H, W)
        :return: output_tensor
        """
        output_tensor = torch.max(self.cSE(input_tensor), self.sSE(input_tensor))
        return output_tensor


class ProjectExciteLayer(nn.Module):
    """
        Project & Excite Module, specifically designed for 3D inputs
        *quote*
    """

    def __init__(self, num_channels, reduction_ratio=2):
        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ProjectExciteLayer, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.relu = nn.ReLU()
        self.conv_c = nn.Conv3d(in_channels=num_channels, out_channels=num_channels_reduced, kernel_size=1, stride=1)
        self.conv_cT = nn.Conv3d(in_channels=num_channels_reduced, out_channels=num_channels, kernel_size=1, stride=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, D, H, W)
        :return: output tensor
        """
        batch_size, num_channels, D, H, W = input_tensor.size()

        # Project:
        # Average along channels and different axes
        squeeze_tensor_w = F.adaptive_avg_pool3d(input_tensor, (1, 1, W))

        squeeze_tensor_h = F.adaptive_avg_pool3d(input_tensor, (1, H, 1))

        squeeze_tensor_d = F.adaptive_avg_pool3d(input_tensor, (D, 1, 1))

        # tile tensors to original size and add:
        final_squeeze_tensor = sum([squeeze_tensor_w.view(batch_size, num_channels, 1, 1, W),
                                    squeeze_tensor_h.view(batch_size, num_channels, 1, H, 1),
                                    squeeze_tensor_d.view(batch_size, num_channels, D, 1, 1)])

        # Excitation:
        final_squeeze_tensor = self.sigmoid(self.conv_cT(self.relu(self.conv_c(final_squeeze_tensor))))
        output_tensor = torch.mul(input_tensor, final_squeeze_tensor)

        return output_tensor


class SELayer3D(Enum):
    """
    Enum restricting the type of SE Blockes available. So that type checking can be adding when adding these blocks to
    a neural network::
        if self.se_block_type == se.SELayer3D.CSE3D.value:
            self.SELayer = se.ChannelSELayer3D(params['num_filters'])
        elif self.se_block_type == se.SELayer3D.SSE3D.value:
            self.SELayer = se.SpatialSELayer3D(params['num_filters'])
        elif self.se_block_type == se.SELayer3D.CSSE3D.value:
            self.SELayer = se.ChannelSpatialSELayer3D(params['num_filters'])
        elif self.se_block_type == se.SELayer3D.PE.value:
            self.SELayer = se.ProjectExcite(params['num_filters')
    """
    NONE = 'NONE'
    CSE3D = 'CSE3D'
    SSE3D = 'SSE3D'
    CSSE3D = 'CSSE3D'
    PE = 'PE'
# x = torch.rand(1,1,4,4,4)
# print(x)
# batch_size, num_channels, d, h, w = x.size()
# x1 =F.adaptive_avg_pool3d(x,(d,1,w))
# x2 =F.adaptive_avg_pool3d(x,(1,h,1))
# x3 =sum(x1,x2)
# print(x3.shape)

# z4= F.adaptive_avg_pool2d(z1,1)
# print(z4)
# print(x)
# A=ChannelSELayer3D(num_channels=64)
# B=SpatialSELayer3D(num_channels=64)
# C=ProjectExciteLayer(num_channels=64)
# x1=A(x)
# x2=B(x)
# x3=C(x)
# print(x1.shape)
# print(x2.shape)
# print(x3.shape)
# p1=ProjectExciteLayer(x)
# print(p1)

class AttentionPooling(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.cls_vec = nn.Parameter(torch.randn(in_dim))
        self.fc = nn.Linear(in_dim, in_dim)
        self.softmax = nn.Softmax(-1)

    def forward(self, x):
        weights = torch.matmul(x.view(-1, x.shape[1]), self.cls_vec)
        weights = self.softmax(weights.view(x.shape[0], -1))
        x = torch.bmm(x.view(x.shape[0], x.shape[1], -1), weights.unsqueeze(-1)).squeeze()
        x = x + self.cls_vec
        x = self.fc(x)
        x = x + self.cls_vec
        return x

class PE(nn.Module):
    """
        Project & Excite Module, specifically designed for 3D inputs
        *quote*
    """

    def __init__(self, num_channels, reduction_ratio=2):
        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(PE, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.relu = nn.ReLU()
        self.conv_c = nn.Conv3d(in_channels=num_channels, out_channels=num_channels_reduced, kernel_size=1, stride=1)
        self.conv_cT = nn.Conv3d(in_channels=num_channels_reduced, out_channels=num_channels, kernel_size=1, stride=1)
        self.sigmoid = nn.Sigmoid()
        self.pooling = AttentionPooling(in_dim=num_channels)

    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, D, H, W)
        :return: output tensor
        """
        batch_size, num_channels, D, H, W = input_tensor.size()

        # Project:
        # Average along channels and different axes
        squeeze_tensor_w = F.adaptive_avg_pool3d(input_tensor, (1, 1, W))

        squeeze_tensor_h = F.adaptive_avg_pool3d(input_tensor, (1, H, 1))

        squeeze_tensor_d = F.adaptive_avg_pool3d(input_tensor, (D, 1, 1))

        # tile tensors to original size and add:
        final_squeeze_tensor = sum([squeeze_tensor_w.view(batch_size, num_channels, 1, 1, W),
                                    squeeze_tensor_h.view(batch_size, num_channels, 1, H, 1),
                                    squeeze_tensor_d.view(batch_size, num_channels, D, 1, 1)])
        #################################################################

        squeeze_tensor_w1 = F._adaptive_max_pool3d(input_tensor, (1, 1, W))

        squeeze_tensor_h1 = F._adaptive_max_pool3d(input_tensor, (1, H, 1))

        squeeze_tensor_d1 = F._adaptive_max_pool3d(input_tensor, (D, 1, 1))

        final_squeeze_tensor1 = sum([squeeze_tensor_w1.view(batch_size, num_channels, 1, 1, W),
                                    squeeze_tensor_h1.view(batch_size, num_channels, 1, H, 1),
                                    squeeze_tensor_d1.view(batch_size, num_channels, D, 1, 1)])
        # Excitation:
        final_squeeze_tensor = self.conv_cT(self.relu(self.conv_c(final_squeeze_tensor)))

        final_squeeze_tensor1 = self.conv_cT(self.relu(self.conv_c(final_squeeze_tensor1)))

        final_squeeze_tensor2=self.sigmoid(final_squeeze_tensor + final_squeeze_tensor1)
        output_tensor = torch.mul(input_tensor, final_squeeze_tensor2)

        return output_tensor

class PPEE(nn.Module):
    """
        Project & Excite Module, specifically designed for 3D inputs
        *quote*
    """

    def __init__(self, num_channels, reduction_ratio=2):
        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(PPEE, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.relu = nn.ReLU()
        self.conv_c = nn.Conv3d(in_channels=num_channels, out_channels=num_channels_reduced, kernel_size=1, stride=1)
        self.conv_cT = nn.Conv3d(in_channels=num_channels_reduced, out_channels=num_channels, kernel_size=1, stride=1)
        self.conv_c1 = nn.Conv3d(in_channels=num_channels_reduced, out_channels=1, kernel_size=1, stride=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, D, H, W)
        :return: output tensor
        """
        batch_size, num_channels, D, H, W = input_tensor.size()

        i1=self.relu(self.conv_c(input_tensor))
        i2=self.sigmoid(self.conv_c1(i1))
        i3=torch.mul(input_tensor,i2)

        i4=sum([input_tensor,i3])





        # Project:
        # Average along channels and different axes
        squeeze_tensor_w = F.adaptive_avg_pool3d(i4, (1, 1, W))

        squeeze_tensor_h = F.adaptive_avg_pool3d(i4, (1, H, 1))

        squeeze_tensor_d = F.adaptive_avg_pool3d(i4, (D, 1, 1))

        # tile tensors to original size and add:
        final_squeeze_tensor = sum([squeeze_tensor_w.view(batch_size, num_channels, 1, 1, W),
                                    squeeze_tensor_h.view(batch_size, num_channels, 1, H, 1),
                                    squeeze_tensor_d.view(batch_size, num_channels, D, 1, 1)])

        # Excitation:
        final_squeeze_tensor = self.sigmoid(self.conv_cT(self.relu(self.conv_c(final_squeeze_tensor))))
        output_tensor = torch.mul(i4, final_squeeze_tensor)

        return output_tensor


class PEE(nn.Module):
    """
        Project & Excite Module, specifically designed for 3D inputs
        *quote*
    """

    def __init__(self, num_channels, reduction_ratio=2):
        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(PEE, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.relu = nn.ReLU()

        self.conv_c = nn.Conv2d(in_channels=num_channels, out_channels=num_channels_reduced, kernel_size=1, stride=1)
        self.conv_cT = nn.Conv2d(in_channels=num_channels_reduced, out_channels=num_channels, kernel_size=1, stride=1)
        self.sigmoid = nn.Sigmoid()


    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, D, H, W)
        :return: output tensor
        """
        batch_size, num_channels, D, H, W = input_tensor.size()

        # Project:
        # Average along channels and different axes
        squeeze_tensor_w = F.adaptive_avg_pool3d(input_tensor, (D, 1, W))

        squeeze_tensor_h = F.adaptive_avg_pool3d(input_tensor, (1, H, W))

        squeeze_tensor_d = F.adaptive_avg_pool3d(input_tensor, (D, H, 1))

        # tile tensors to original size and add:


        final_squeeze_tensor1 = self.conv_cT(self.relu(self.conv_c(squeeze_tensor_w.view(batch_size, num_channels,D,W))))

        final_squeeze_tensor2 = self.conv_cT(self.relu(self.conv_c(squeeze_tensor_h.view(batch_size, num_channels,H,W))))

        final_squeeze_tensor3 = self.conv_cT(self.relu(self.conv_c(squeeze_tensor_d.view(batch_size, num_channels,D,H))))


        final_squeeze_tensor4 = sum([final_squeeze_tensor1.view(batch_size, num_channels,D,1,W),
                                    final_squeeze_tensor2.view(batch_size, num_channels,1,H,W),
                                    final_squeeze_tensor3.view(batch_size, num_channels,D,H,1)])
        output_tensor = torch.mul(input_tensor, final_squeeze_tensor4)

        return output_tensor

# a = torch.tensor([[1,2],[3,4]])
#
# x1 = torch.tensor([[1,1],[2,2]])
# print(a.shape)
# print(x1.shape)
# print(a)
# print(x1)
# print(sum(a,x1))

# # batch_size, num_channels, d, h, w = x.size()
# # x1 =F.adaptive_avg_pool3d(x,(1,h,w))
# # cv=nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, stride=1)
# # print(x1.shape)
# # x2=cv(x1.view(batch_size, num_channels,4,4))
# C=PEE(num_channels=64)
# print(C(x).shape)
#
# batch_size, num_channels, d, h, w = x.size()
# x1 =F.adaptive_avg_pool3d(x,(1,h,w))
# print(x1.shape)
# cv=nn.Conv3d(in_channels=32, out_channels=16, kernel_size=1, stride=1)
# x2=cv(x1)
# print(x2.shape)
# x2 =F.adaptive_avg_pool3d(x,(1,h,1))
# x3 =sum(x1,x2)
# print(x3.shape)

# z4= F.adaptive_avg_pool2d(z1,1)
# print(z4)
# print(x)
# A=ChannelSELayer3D(num_channels=64)
# B=SpatialSELayer3D(num_channels=64)
# C=ProjectExciteLayer(num_channels=64)
# x1=A(x)
# x2=B(x)
# x3=C(x)
# print(x1.shape)
# print(x2.shape)
# print(x3.shape)
# p1=ProjectExciteLayer(x)
# print(p1)


class PCA(nn.Module):
    """
        Project & Excite Module, specifically designed for 3D inputs
        *quote*
    """

    def __init__(self, num_channel, reduction_ratio=2):
        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(PCA, self).__init__()
        num_channels_reduced = num_channel // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.relu = nn.ReLU()
        self.conv_c = nn.Conv3d(in_channels=num_channel, out_channels=num_channels_reduced, kernel_size=1, stride=1)
        self.conv_cT = nn.Conv3d(in_channels=num_channels_reduced, out_channels=num_channel, kernel_size=1, stride=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, g,x):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, D, H, W)
        :return: output tensor
        """
        batch_size, num_channels, D, H, W = g.size()
        batch_size1, num_channels1, D1, H1, W1 = x.size()

        # Project:
        # Average along channels and different axes
        squeeze_tensor_w = F.adaptive_avg_pool3d(g, (1, 1, W))

        squeeze_tensor_h = F.adaptive_avg_pool3d(g, (1, H, 1))

        squeeze_tensor_d = F.adaptive_avg_pool3d(g, (D, 1, 1))

        # tile tensors to original size and add:
        final_squeeze_tensor = sum([squeeze_tensor_w.view(batch_size, num_channels, 1, 1, W),
                                    squeeze_tensor_h.view(batch_size, num_channels, 1, H, 1),
                                    squeeze_tensor_d.view(batch_size, num_channels, D, 1, 1)])
        #################################################################

        squeeze_tensor_w1 = F.adaptive_avg_pool3d(x, (1, 1, W1))

        squeeze_tensor_h1 = F.adaptive_avg_pool3d(x, (1, H1, 1))

        squeeze_tensor_d1 = F.adaptive_avg_pool3d(x, (D1, 1, 1))

        final_squeeze_tensor1 = sum([squeeze_tensor_w1.view(batch_size1, num_channels1, 1, 1, W1),
                                     squeeze_tensor_h1.view(batch_size1, num_channels1, 1, H1, 1),
                                     squeeze_tensor_d1.view(batch_size1, num_channels1, D1, 1, 1)])
        # Excitation:
        final_squeeze_tensor = self.conv_cT(self.relu(self.conv_c(final_squeeze_tensor)))

        final_squeeze_tensor1 = self.conv_cT(self.relu(self.conv_c(final_squeeze_tensor1)))

        final_squeeze_tensor2 = self.sigmoid(final_squeeze_tensor + final_squeeze_tensor1)
        output_tensor = torch.mul(x, final_squeeze_tensor2)



        return output_tensor
