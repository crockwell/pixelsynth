import math

import numpy as np
import torch 
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils import weight_norm as wn
from torch.utils.checkpoint import checkpoint

from models.lmconv.layers import *
from models.lmconv.locally_masked_convolution import locally_masked_conv2d
from models.lmconv.utils import *


class OurPixelCNNLayer_up(nn.Module):
    def __init__(self, nr_resnet, nr_filters, resnet_nonlinearity, conv_op, feature_norm_op=None,
                 kernel_size=(5,5), weight_norm=True, dropout_prob=0.5, rematerialize=False):
        super(OurPixelCNNLayer_up, self).__init__()
        self.nr_resnet = nr_resnet
        self.u_stream = nn.ModuleList([gated_resnet(nr_filters, conv_op, feature_norm_op,
                                        resnet_nonlinearity, skip_connection=0, dropout_prob=dropout_prob) 
                                            for _ in range(nr_resnet)])

        self.rematerialize = rematerialize
        
    def forward(self, u, mask=None):
        u_list = []
        for i in range(self.nr_resnet):
            if self.rematerialize:
                u  = checkpoint(self.u_stream[i], u, None, mask)
            else:
                u  = self.u_stream[i](u, mask=mask)
            u_list  += [u]
        return u_list


class OurPixelCNNLayer_down(nn.Module):
    def __init__(self, nr_resnet, nr_filters, resnet_nonlinearity, conv_op, feature_norm_op=None,
                 kernel_size=(5,5), weight_norm=True, dropout_prob=0.5, rematerialize=False):
        super(OurPixelCNNLayer_down, self).__init__()
        self.nr_resnet = nr_resnet

        self.u_stream  = nn.ModuleList([gated_resnet(nr_filters, conv_op, feature_norm_op,
                                        resnet_nonlinearity, skip_connection=1, dropout_prob=dropout_prob) 
                                            for _ in range(nr_resnet)])

        self.rematerialize = rematerialize

    def forward(self, u, u_list, mask=None):
        for i in range(self.nr_resnet):
            a = u_list.pop()
            if self.rematerialize:
                u  = checkpoint(self.u_stream[i], u, a, mask)
            else:
                u  = self.u_stream[i](u, a=a, mask=mask)
        return u


class OurPixelCNN(nn.Module):
    def __init__(self, nr_resnet=5, nr_filters=80, nr_logistic_mix=10,
                    resnet_nonlinearity='concat_elu', input_channels=3, kernel_size=(5,5),
                    max_dilation=2, weight_norm=True, feature_norm_op=None, dropout_prob=0.5, conv_bias=True,
                    conv_mask_weight=False, rematerialize=False, binarize=False):
        super(OurPixelCNN, self).__init__()
        assert resnet_nonlinearity == 'concat_elu'
        self.resnet_nonlinearity = lambda x : concat_elu(x)
        self.init_padding = None
        self.binarize = binarize

        if weight_norm:
            conv_op_init = lambda cin, cout: wn(locally_masked_conv2d(cin, cout, kernel_size=kernel_size, bias=conv_bias, mask_weight=conv_mask_weight))
            conv_op_dilated = lambda cin, cout: wn(locally_masked_conv2d(cin, cout, kernel_size=kernel_size, dilation=max_dilation, bias=conv_bias, mask_weight=conv_mask_weight))
            conv_op = lambda cin, cout: wn(locally_masked_conv2d(cin, cout, kernel_size=kernel_size, bias=conv_bias, mask_weight=conv_mask_weight))
        else:
            conv_op_init = lambda cin, cout: locally_masked_conv2d(cin, cout, kernel_size=kernel_size, bias=conv_bias, mask_weight=conv_mask_weight)
            conv_op_dilated = lambda cin, cout: locally_masked_conv2d(cin, cout, kernel_size=kernel_size, dilation=max_dilation, bias=conv_bias, mask_weight=conv_mask_weight)
            conv_op = lambda cin, cout: locally_masked_conv2d(cin, cout, kernel_size=kernel_size, bias=conv_bias, mask_weight=conv_mask_weight)

        down_nr_resnet = [nr_resnet] + [nr_resnet + 1] * 2
        self.down_layers = nn.ModuleList([OurPixelCNNLayer_down(down_nr_resnet[i], nr_filters, self.resnet_nonlinearity, conv_op,
                                                feature_norm_op, kernel_size=kernel_size, weight_norm=weight_norm,
                                                dropout_prob=dropout_prob, rematerialize=rematerialize) for i in range(3)])

        self.up_layers = nn.ModuleList([OurPixelCNNLayer_up(nr_resnet, nr_filters, self.resnet_nonlinearity, conv_op,
                                                feature_norm_op, kernel_size=kernel_size, weight_norm=weight_norm,
                                                dropout_prob=dropout_prob, rematerialize=rematerialize) for _ in range(3)])

        self.u_init = conv_op_init(input_channels + 1, nr_filters)
        self.downsize_u_stream = nn.ModuleList([conv_op_dilated(nr_filters, nr_filters) for _ in range(2)])
        self.upsize_u_stream = nn.ModuleList([conv_op_dilated(nr_filters, nr_filters) for _ in range(2)])

        self.norm_init = feature_norm_op(nr_filters) if feature_norm_op else identity
        self.norm_ds = nn.ModuleList([feature_norm_op(nr_filters) for _ in range(2)]) if feature_norm_op else None
        self.norm_us = nn.ModuleList([feature_norm_op(nr_filters) for _ in range(2)]) if feature_norm_op else None

        if self.binarize:
            self.nin_out = nin(nr_filters, 2, weight_norm=True)
        else:
            num_mix = 3 if input_channels == 1 else 10
            if input_channels == 4:
                num_mix = 12
                nr_logistic_mix = 13
            elif input_channels == 6:
                num_mix = 31
                nr_logistic_mix = 12
            self.nin_out = nin(nr_filters, 512, weight_norm=True)

    def forward(self, x, sample=False, mask_init=None, mask_undilated=None, mask_dilated=None):
        # similar as done in the tf repo. remake init_padding if input height or width change :  
        if isinstance(x, list):
            mask_init = x[1]
            mask_undilated = x[2]
            mask_dilated = x[3]
            x = x[0]

        if not sample and (self.init_padding is None or self.init_padding.shape[2:] != x.shape[2:]) :
            xs = [int(y) for y in x.size()]
            padding = Variable(torch.ones(xs[0], 1, xs[2], xs[3]), requires_grad=False)
            self.init_padding = padding.cuda() if x.is_cuda else padding

        if sample : 
            xs = [int(y) for y in x.size()]
            padding = Variable(torch.ones(xs[0], 1, xs[2], xs[3]), requires_grad=False)
            padding = padding.cuda() if x.is_cuda else padding
            x = torch.cat((x, padding), 1)
        
        x = x if sample else torch.cat((x, self.init_padding), 1)

        ###      UP PASS    ###
        u_list  = [self.norm_init(self.u_init(x, mask=mask_init), mask=mask_undilated)]  # mask_undilated passed intentionally
                                                                                         # to norm so center pixel is 1,
                                                                                         # preventing division by zero for OrderRescale
        # resnet block and dilation (RENAME: does not downsize)
        for i in range(2):
            u_list += self.up_layers[i](u_list[-1], mask=mask_undilated)
            u_list += [self.downsize_u_stream[i](u_list[-1], mask=mask_dilated)]
            if self.norm_ds:
                u_list[-1] = self.norm_ds[i](u_list[-1], mask=mask_dilated)
        u_list += self.up_layers[2](u_list[-1], mask=mask_undilated)

        ###    DOWN PASS    ###
        # resnet block and dilation (RENAME: does not upsize)
        u = u_list.pop()
        for i in range(2):
            u = self.down_layers[i](u, u_list, mask=mask_undilated)
            u = self.upsize_u_stream[i](u, mask=mask_dilated)
            if self.norm_us:
                u = self.norm_us[i](u, mask=mask_dilated)
        u = self.down_layers[2](u, u_list, mask=mask_undilated)

        x_out = self.nin_out(F.elu(u))

        return x_out
