import math

import torch 
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F


class _locally_masked_conv2d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, mask, weight, mask_weight=None, bias=None, dilation=1, padding=1):
        assert len(x.shape) == 4, "Unfold/fold only support 4D batched image-like tensors"
        ctx.save_for_backward(x, mask, weight, mask_weight)
        ctx.dilation = dilation
        ctx.padding = padding
        ctx.H, ctx.W = x.size(2), x.size(3)

        # Shapes
        ctx.output_shape = (x.shape[2], x.shape[3])
        out_channels, in_channels, k1, k2 = weight.shape
        assert x.size(1) == in_channels
        assert mask.size(1) == k1 * k2

        # Step 1: Unfold (im2col)
        x = F.unfold(x, (k1, k2), dilation=dilation, padding=padding)

        # Step 2: Mask x. Avoid repeating mask in_channels
        #         times by reshaping x_unf (memory efficient)
        assert x.size(1) % in_channels == 0
        x_unf_channels_batched = x.view(x.size(0) * in_channels,
                                        x.size(1) // in_channels,
                                        x.size(2))
        #import pdb 
        #pdb.set_trace()
        #print(x_unf_channels_batched.shape, mask.shape)
        x = torch.mul(x_unf_channels_batched, mask).view(x.shape)

        # Step 3: Perform convolution via matrix multiplication and addition
        weight_matrix = weight.view(out_channels, -1)
        x = weight_matrix.matmul(x)
        if bias is not None:
            x = x + bias.unsqueeze(0).unsqueeze(2)

        # Step 4: Apply weight on mask, if provided. Equivalent to concatenating x and mask.
        if mask_weight is not None:
            x = x + mask_weight.view(out_channels, -1).matmul(mask)

        # Step 4: Restore shape
        output = x.view(x.size(0), x.size(1), *ctx.output_shape)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, mask, weight, mask_weight = ctx.saved_tensors
        out_channels, in_channels, k1, k2 = weight.shape
        grad_output_unfolded = grad_output.view(grad_output.size(0),
                                                grad_output.size(1),
                                                -1)  # B x C_out x (H*W)

        # Compute gradients
        grad_x = grad_weight = grad_mask_weight = grad_bias = None
        if ctx.needs_input_grad[0]:
            weight_ = weight.view(out_channels, -1)
            grad_x_ = weight_.transpose(0, 1).matmul(grad_output_unfolded)
            grad_x_shape = grad_x_.shape
            # View to allow masking, since mask needs to be broadcast C_in times
            assert grad_x_.size(1) % in_channels == 0
            grad_x_ = grad_x_.view(grad_x_.size(0) * in_channels,
                                   grad_x_.size(1) // in_channels,
                                   grad_x_.size(2))
            grad_x_ = torch.mul(grad_x_, mask).view(grad_x_shape)
            grad_x = F.fold(grad_x_, (ctx.H, ctx.W), (k1, k2), dilation=ctx.dilation, padding=ctx.padding)
        if ctx.needs_input_grad[2]:
            # Recompute unfold and masking to avoid storing unfolded x, at the cost of extra compute
            x_ = F.unfold(x, (k1, k2), dilation=ctx.dilation, padding=ctx.padding)  # B x 27 x 64
            x_unf_shape = x_.shape
            assert x_.size(1) % in_channels == 0
            x_ = x_.view(x_.size(0) * in_channels,
                         x_.size(1) // in_channels,
                         x_.size(2))
            x_ = torch.mul(x_, mask).view(x_unf_shape)

            grad_weight = grad_output_unfolded.matmul(x_.transpose(2, 1))
            grad_weight = grad_weight.view(grad_weight.size(0), *weight.shape)
        if ctx.needs_input_grad[3]:
            grad_mask_weight = grad_output_unfolded.matmul(mask.transpose(2, 1))  # B x C_out x k1*k2
            grad_mask_weight = grad_mask_weight.view(grad_mask_weight.size(0), *mask_weight.shape)
        if ctx.needs_input_grad[4]:
            grad_bias = grad_output.sum(dim=(0, 2, 3))

        assert not ctx.needs_input_grad[1], "Can't differentiate wrt mask"

        return grad_x, None, grad_weight, grad_mask_weight, grad_bias, None, None


class locally_masked_conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3,3), dilation=1, bias=True, mask_weight=False):
        """A memory-efficient implementation of Locally Masked Convolution.

        Args:
            in_channels (int): Number of channels in the input image
            out_channels (int): Number of channels produced by the convolution
            kernel_size (tuple): Size of the convolving kernel as a tuple of two ints.
                Default: (3, 3). The first int is used for the height dimension,
                and the second int for the width dimension.
            dilation (int): Spacing between kernel elements. Default: 1
            bias (bool): If True, adds a learnable bias to the output. Default: True
            mask_weight (bool): If True, adds a learnable weight to condition the layer
                on the mask. Default: False
        """
        super(locally_masked_conv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dilation = dilation

        # Pad to maintain spatial dimensions
        pad0 = (dilation * (kernel_size[0] - 1)) // 2
        pad1 = (dilation * (kernel_size[1] - 1)) // 2
        self.padding = (pad0, pad1)

        # Conv parameters
        self.weight = Parameter(torch.Tensor(out_channels, in_channels, *kernel_size))
        self.mask_weight = Parameter(torch.Tensor(out_channels, *kernel_size)) if mask_weight else None
        self.bias = Parameter(torch.Tensor(out_channels)) if bias else None

        self.reset_parameters()

    def reset_parameters(self):
        # Adapted from PyTorch _ConvNd implementation
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.mask_weight is not None:
            nn.init.kaiming_uniform_(self.mask_weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x, mask=None):
        return _locally_masked_conv2d.apply(x, mask, self.weight, self.mask_weight, self.bias,
                                          self.dilation, self.padding)
