from pytorch_wavelets import DTCWTForward, DTCWTInverse

from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.ans import BufferedRansEncoder, RansDecoder
from compressai.models import CompressionModel
from compressai.layers import (
    AttentionBlock,
    ResidualBlock,
    ResidualBlockUpsample,
    ResidualBlockWithStride,
    conv3x3,
    subpel_conv3x3, GDN,
)

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from timm.layers import trunc_normal_, DropPath
import numpy as np
import math
import time
import os
import matplotlib.pyplot as plt

SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64
import pywt
from torch.autograd import Function


class DWT_Function(Function):
    @staticmethod
    def forward(ctx, x, w_ll, w_lh, w_hl, w_hh):
        x = x.contiguous()
        ctx.save_for_backward(w_ll, w_lh, w_hl, w_hh)
        ctx.shape = x.shape

        dim = x.shape[1]
        x_ll = torch.nn.functional.conv2d(x, w_ll.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x_lh = torch.nn.functional.conv2d(x, w_lh.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x_hl = torch.nn.functional.conv2d(x, w_hl.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x_hh = torch.nn.functional.conv2d(x, w_hh.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x = torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1)
        return x

    @staticmethod
    def backward(ctx, dx):
        if ctx.needs_input_grad[0]:
            w_ll, w_lh, w_hl, w_hh = ctx.saved_tensors
            B, C, H, W = ctx.shape
            dx = dx.view(B, 4, -1, H // 2, W // 2)

            dx = dx.transpose(1, 2).reshape(B, -1, H // 2, W // 2)
            filters = torch.cat([w_ll, w_lh, w_hl, w_hh], dim=0).repeat(C, 1, 1, 1)
            dx = torch.nn.functional.conv_transpose2d(dx, filters, stride=2, groups=C)

        return dx, None, None, None, None


class IDWT_Function(Function):
    @staticmethod
    def forward(ctx, x, filters):
        ctx.save_for_backward(filters)
        ctx.shape = x.shape

        B, _, H, W = x.shape
        x = x.view(B, 4, -1, H, W).transpose(1, 2)
        C = x.shape[1]
        x = x.reshape(B, -1, H, W)
        filters = filters.repeat(C, 1, 1, 1)
        x = torch.nn.functional.conv_transpose2d(x, filters, stride=2, groups=C)
        return x

    @staticmethod
    def backward(ctx, dx):
        if ctx.needs_input_grad[0]:
            filters = ctx.saved_tensors
            filters = filters[0]
            B, C, H, W = ctx.shape
            C = C // 4
            dx = dx.contiguous()

            w_ll, w_lh, w_hl, w_hh = torch.unbind(filters, dim=0)
            x_ll = torch.nn.functional.conv2d(dx, w_ll.unsqueeze(1).expand(C, -1, -1, -1), stride=2, groups=C)
            x_lh = torch.nn.functional.conv2d(dx, w_lh.unsqueeze(1).expand(C, -1, -1, -1), stride=2, groups=C)
            x_hl = torch.nn.functional.conv2d(dx, w_hl.unsqueeze(1).expand(C, -1, -1, -1), stride=2, groups=C)
            x_hh = torch.nn.functional.conv2d(dx, w_hh.unsqueeze(1).expand(C, -1, -1, -1), stride=2, groups=C)
            dx = torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1)
        return dx, None


class IDWT_2D(nn.Module):
    def __init__(self, wave):
        super(IDWT_2D, self).__init__()
        w = pywt.Wavelet(wave)
        rec_hi = torch.Tensor(w.rec_hi)
        rec_lo = torch.Tensor(w.rec_lo)

        w_ll = rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1)
        w_lh = rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1)
        w_hl = rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1)
        w_hh = rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)

        w_ll = w_ll.unsqueeze(0).unsqueeze(1)
        w_lh = w_lh.unsqueeze(0).unsqueeze(1)
        w_hl = w_hl.unsqueeze(0).unsqueeze(1)
        w_hh = w_hh.unsqueeze(0).unsqueeze(1)
        filters = torch.cat([w_ll, w_lh, w_hl, w_hh], dim=0)
        self.register_buffer('filters', filters)
        # self.filters = self.filters.to(dtype=torch.float16)
        self.filters = self.filters.to(dtype=torch.float32)

    def forward(self, x):
        return IDWT_Function.apply(x, self.filters)


class DWT_2D(nn.Module):
    def __init__(self, wave):
        super(DWT_2D, self).__init__()
        w = pywt.Wavelet(wave)
        dec_hi = torch.Tensor(w.dec_hi[::-1])
        dec_lo = torch.Tensor(w.dec_lo[::-1])

        w_ll = dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1)
        w_lh = dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1)
        w_hl = dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1)
        w_hh = dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)

        self.register_buffer('w_ll', w_ll.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_lh', w_lh.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_hl', w_hl.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_hh', w_hh.unsqueeze(0).unsqueeze(0))

        # self.w_ll = self.w_ll.to(dtype=torch.float16)
        # self.w_lh = self.w_lh.to(dtype=torch.float16)
        # self.w_hl = self.w_hl.to(dtype=torch.float16)
        # self.w_hh = self.w_hh.to(dtype=torch.float16)

        self.w_ll = self.w_ll.to(dtype=torch.float32)
        self.w_lh = self.w_lh.to(dtype=torch.float32)
        self.w_hl = self.w_hl.to(dtype=torch.float32)
        self.w_hh = self.w_hh.to(dtype=torch.float32)

    def forward(self, x):
        return DWT_Function.apply(x, self.w_ll, self.w_lh, self.w_hl, self.w_hh)


def conv1x1(in_ch: int, out_ch: int, stride: int = 1) -> nn.Module:
    """1x1 convolution."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride)


def conv(in_channels, out_channels, kernel_size=5, stride=2):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=kernel_size // 2,
    )


def deconv(in_channels, out_channels, kernel_size=5, stride=2):
    return nn.ConvTranspose2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        output_padding=stride - 1,
        padding=kernel_size // 2,
    )


def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))


def ste_round(x: Tensor) -> Tensor:
    return torch.round(x) - x.detach() + x


def find_named_module(module, query):
    """Helper function to find a named module. Returns a `nn.Module` or `None`

    Args:
        module (nn.Module): the root module
        query (str): the module name to find

    Returns:
        nn.Module or None
    """

    return next((m for n, m in module.named_modules() if n == query), None)


def find_named_buffer(module, query):
    """Helper function to find a named buffer. Returns a `torch.Tensor` or `None`

    Args:
        module (nn.Module): the root module
        query (str): the buffer name to find

    Returns:
        torch.Tensor or None
    """
    return next((b for n, b in module.named_buffers() if n == query), None)


def _update_registered_buffer(
        module,
        buffer_name,
        state_dict_key,
        state_dict,
        policy="resize_if_empty",
        dtype=torch.int,
):
    # state_dict_key = state_dict if state_dict_key in state_dict.keys() else "module." + state_dict_key

    new_size = state_dict[state_dict_key].size()
    registered_buf = find_named_buffer(module, buffer_name)

    if policy in ("resize_if_empty", "resize"):  # resize
        if registered_buf is None:
            raise RuntimeError(f'buffer "{buffer_name}" was not registered')

        if policy == "resize" or registered_buf.numel() == 0:
            registered_buf.resize_(new_size)

    elif policy == "register":
        if registered_buf is not None:
            raise RuntimeError(f'buffer "{buffer_name}" was already registered')

        module.register_buffer(buffer_name, torch.empty(new_size, dtype=dtype).fill_(0))

    else:
        raise ValueError(f'Invalid policy "{policy}"')


def update_registered_buffers(
        module,
        module_name,
        buffer_names,
        state_dict,
        policy="resize_if_empty",
        dtype=torch.int,
):
    """Update the registered buffers in a module according to the tensors sized
    in a state_dict.
    (There's no way in torch to directly load a buffer with a dynamic size)

    Args:
        module (nn.Module): the module
        module_name (str): module name in the state dict
        buffer_names (list(str)): list of the buffer names to resize in the module
        state_dict (dict): the state dict
        policy (str): Update policy, choose from
            ('resize_if_empty', 'resize', 'register')
        dtype (dtype): Type of buffer to be registered (when policy is 'register')
    """
    if not module:
        return
    valid_buffer_names = [n for n, _ in module.named_buffers()]
    for buffer_name in buffer_names:
        if buffer_name not in valid_buffer_names:
            raise ValueError(f'Invalid buffer name "{buffer_name}"')

    for buffer_name in buffer_names:
        _update_registered_buffer(
            module,
            buffer_name,
            f"{module_name}.{buffer_name}",
            state_dict,
            policy,
            dtype,
        )


class ResidualBottleneckBlock(nn.Module):
    """Residual bottleneck block.

    Introduced by [He2016], this block sandwiches a 3x3 convolution
    between two 1x1 convolutions which reduce and then restore the
    number of channels. This reduces the number of parameters required.

    [He2016]: `"Deep Residual Learning for Image Recognition"
    <https://arxiv.org/abs/1512.03385>`_, by Kaiming He, Xiangyu Zhang,
    Shaoqing Ren, and Jian Sun, CVPR 2016.

    Args:
        in_ch (int): Number of input channels
        out_ch (int): Number of output channels
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        mid_ch = min(in_ch, out_ch) // 2
        self.conv1 = conv1x1(in_ch, mid_ch)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(mid_ch, mid_ch)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = conv1x1(mid_ch, out_ch)
        self.skip = conv1x1(in_ch, out_ch) if in_ch != out_ch else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        identity = self.skip(x)

        out = x
        out = self.conv1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)

        return out + identity


class ResidualBottleneckBlockWithStride(nn.Module):

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = ResidualBlockWithStride_wave(in_ch, out_ch)
        self.res1 = ResidualBottleneckBlock(out_ch, out_ch)
        self.res2 = ResidualBottleneckBlock(out_ch, out_ch)
        self.res3 = ResidualBottleneckBlock(out_ch, out_ch)

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv(x)
        out = self.res1(out)
        out = self.res2(out)
        out = self.res3(out)

        return out


class ResidualBottleneckBlockWithStride_no(nn.Module):

    def __init__(self, in_ch: int, out_ch: int, stride=1):
        super().__init__()
        self.conv = conv(in_ch, out_ch, kernel_size=5, stride=stride)
        self.res1 = ResidualBottleneckBlock(out_ch, out_ch)
        self.res2 = ResidualBottleneckBlock(out_ch, out_ch)
        self.res3 = ResidualBottleneckBlock(out_ch, out_ch)

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv(x)
        out = self.res1(out)
        out = self.res2(out)
        out = self.res3(out)

        return out


class ResidualBottleneckBlockWithUpsample(nn.Module):

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.res1 = ResidualBottleneckBlock(in_ch, in_ch)
        self.res2 = ResidualBottleneckBlock(in_ch, in_ch)
        self.res3 = ResidualBottleneckBlock(in_ch, in_ch)
        self.conv = ResidualBlockUpsample_wave(in_ch, out_ch)

    def forward(self, x: Tensor) -> Tensor:
        out = self.res1(x)
        out = self.res2(out)
        out = self.res3(out)
        out = self.conv(out)

        return out


class ResidualBottleneckBlockWithUpsample_no(nn.Module):

    def __init__(self, in_ch: int, out_ch: int, stride=1):
        super().__init__()
        self.res1 = ResidualBottleneckBlock(in_ch, in_ch)
        self.res2 = ResidualBottleneckBlock(in_ch, in_ch)
        self.res3 = ResidualBottleneckBlock(in_ch, in_ch)
        self.conv = deconv(in_ch, out_ch, kernel_size=5, stride=stride)

    def forward(self, x: Tensor) -> Tensor:
        out = self.res1(x)
        out = self.res2(out)
        out = self.res3(out)
        out = self.conv(out)

        return out


class ResidualBlockWithStride_wave(nn.Module):
    """Residual block with a stride on the first convolution.

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
        stride (int): stride value (default: 2)
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 2, wavelet='haar'):
        super().__init__()
        self.conv1 = conv3x3(in_ch, out_ch, stride=stride)
        self.leaky_relu = nn.LeakyReLU(inplace=True)

        self.gdn_low = GDN(out_ch)
        self.low_freq_conv = conv3x3(out_ch, out_ch)

        self.conv1_new = conv1x1(2*out_ch, 2*out_ch)
        self.conv2_new = conv1x1(2*out_ch, 2*out_ch)
        self.conv3 = conv1x1(2*out_ch, 2*out_ch)
        self.conv4 = conv1x1(2*out_ch, 2*out_ch)
        self.conv5 = conv1x1(2*out_ch, 2*out_ch)
        self.conv6 = conv1x1(2*out_ch, 2*out_ch)

        #添加0-12的gdn
        self.gdn1 = GDN(2*out_ch)
        self.gdn2 = GDN(2*out_ch)
        self.gdn3 = GDN(2*out_ch)
        self.gdn4 = GDN(2*out_ch)
        self.gdn5 = GDN(2*out_ch)
        self.gdn6 = GDN(2*out_ch)

        if stride != 1 or in_ch != out_ch:
            self.skip = conv1x1(in_ch, out_ch, stride=stride)
        else:
            self.skip = None

        self.dtcwt = DTCWTForward(J=1, biort='near_sym_b', qshift='qshift_b').cuda()
        self.idtcwt = DTCWTInverse(biort='near_sym_b', qshift='qshift_b').cuda()

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.conv1(x)
        out = self.leaky_relu(out)
        # DWT
        #dwt_output = self.xfm(out)
        dtcwt_output_l, dtcwt_output_h = self.dtcwt(out)

        low_freq = dtcwt_output_l

        a1 = dtcwt_output_h[0][:, :, 0, :, :, 0]
        a2 = dtcwt_output_h[0][:, :, 1, :, :, 0]
        a3 = dtcwt_output_h[0][:, :, 2, :, :, 0]
        a4 = dtcwt_output_h[0][:, :, 3, :, :, 0]
        a5 = dtcwt_output_h[0][:, :, 4, :, :, 0]
        a6 = dtcwt_output_h[0][:, :, 5, :, :, 0]

        b1 = dtcwt_output_h[0][:, :, 0, :, :, 1]
        b2 = dtcwt_output_h[0][:, :, 1, :, :, 1]
        b3 = dtcwt_output_h[0][:, :, 2, :, :, 1]
        b4 = dtcwt_output_h[0][:, :, 3, :, :, 1]
        b5 = dtcwt_output_h[0][:, :, 4, :, :, 1]
        b6 = dtcwt_output_h[0][:, :, 5, :, :, 1]

        c1=torch.cat([a1,b1],dim=1)
        c2=torch.cat([a2,b2],dim=1)
        c3=torch.cat([a3,b3],dim=1)
        c4=torch.cat([a4,b4],dim=1)
        c5=torch.cat([a5,b5],dim=1)
        c6=torch.cat([a6,b6],dim=1)


        new_y_outputl = torch.zeros_like(dtcwt_output_l)
        new_y_outputh = torch.zeros_like(dtcwt_output_h[0])


        # Process low-frequency and high-frequency components separately
        low_freq_processed = self.low_freq_conv(low_freq)
        low_freq_processed = self.gdn_low(low_freq_processed)

        c1_processed = self.conv1_new(c1)
        c1_processed = self.gdn1(c1_processed)
        c2_processed = self.conv2_new(c2)
        c2_processed = self.gdn2(c2_processed)
        c3_processed = self.conv3(c3)
        c3_processed = self.gdn3(c3_processed)
        c4_processed = self.conv4(c4)
        c4_processed = self.gdn4(c4_processed)
        c5_processed = self.conv5(c5)
        c5_processed = self.gdn5(c5_processed)
        c6_processed = self.conv6(c6)
        c6_processed = self.gdn6(c6_processed)

        a1_processed,b1_processed=torch.chunk(c1_processed, 2, dim=1)
        a2_processed,b2_processed=torch.chunk(c2_processed, 2, dim=1)
        a3_processed,b3_processed=torch.chunk(c3_processed, 2, dim=1)
        a4_processed,b4_processed=torch.chunk(c4_processed, 2, dim=1)
        a5_processed,b5_processed=torch.chunk(c5_processed, 2, dim=1)
        a6_processed,b6_processed=torch.chunk(c6_processed, 2, dim=1)

        new_y_outputh[:, :, 0, :, :, 0] = a1_processed
        new_y_outputh[:, :, 1, :, :, 0] = a2_processed
        new_y_outputh[:, :, 2, :, :, 0] = a3_processed
        new_y_outputh[:, :, 3, :, :, 0] = a4_processed
        new_y_outputh[:, :, 4, :, :, 0] = a5_processed
        new_y_outputh[:, :, 5, :, :, 0] = a6_processed
        new_y_outputh[:, :, 0, :, :, 1] = b1_processed
        new_y_outputh[:, :, 1, :, :, 1] = b2_processed
        new_y_outputh[:, :, 2, :, :, 1] = b3_processed
        new_y_outputh[:, :, 3, :, :, 1] = b4_processed
        new_y_outputh[:, :, 4, :, :, 1] = b5_processed
        new_y_outputh[:, :, 5, :, :, 1] = b6_processed


        new_y_outputh_list=[]
        new_y_outputh_list.append(new_y_outputh)
        new_y_outputl=low_freq_processed

        # IDWT
        output = self.idtcwt((new_y_outputl, new_y_outputh_list))

        if self.skip is not None:
            identity = self.skip(x)

        output += identity
        return output

class ResidualBlockUpsample_wave(nn.Module):
    """Residual block with sub-pixel upsampling on the last convolution.

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
        upsample (int): upsampling factor (default: 2)
    """

    def __init__(self, in_ch: int, out_ch: int, upsample: int = 2, wavelet='haar'):
        super().__init__()
        self.subpel_conv = subpel_conv3x3(in_ch, out_ch, upsample)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.igdn_low = GDN(out_ch, inverse=True)
        self.upsample = subpel_conv3x3(in_ch, out_ch, upsample)
        self.low_freq_conv = conv3x3(out_ch, out_ch)
        self.conv1_new = conv1x1(2*out_ch, 2*out_ch)
        self.conv2_new = conv1x1(2*out_ch, 2*out_ch)
        self.conv3 = conv1x1(2*out_ch, 2*out_ch)
        self.conv4 = conv1x1(2*out_ch, 2*out_ch)
        self.conv5 = conv1x1(2*out_ch, 2*out_ch)
        self.conv6 = conv1x1(2*out_ch, 2*out_ch)

        #添加0-12的gdn
        self.gdn1 = GDN(2*out_ch, inverse=True)
        self.gdn2 = GDN(2*out_ch, inverse=True)
        self.gdn3 = GDN(2*out_ch, inverse=True)
        self.gdn4 = GDN(2*out_ch, inverse=True)
        self.gdn5 = GDN(2*out_ch, inverse=True)
        self.gdn6 = GDN(2*out_ch, inverse=True)



        self.dtcwt = DTCWTForward(J=1, biort='near_sym_b', qshift='qshift_b').cuda()
        self.idtcwt = DTCWTInverse(biort='near_sym_b', qshift='qshift_b').cuda()

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.subpel_conv(x)
        out = self.leaky_relu(out)

        dtcwt_output_l, dtcwt_output_h = self.dtcwt(out)

        low_freq = dtcwt_output_l

        a1 = dtcwt_output_h[0][:, :, 0, :, :, 0]
        a2 = dtcwt_output_h[0][:, :, 1, :, :, 0]
        a3 = dtcwt_output_h[0][:, :, 2, :, :, 0]
        a4 = dtcwt_output_h[0][:, :, 3, :, :, 0]
        a5 = dtcwt_output_h[0][:, :, 4, :, :, 0]
        a6 = dtcwt_output_h[0][:, :, 5, :, :, 0]

        b1 = dtcwt_output_h[0][:, :, 0, :, :, 1]
        b2 = dtcwt_output_h[0][:, :, 1, :, :, 1]
        b3 = dtcwt_output_h[0][:, :, 2, :, :, 1]
        b4 = dtcwt_output_h[0][:, :, 3, :, :, 1]
        b5 = dtcwt_output_h[0][:, :, 4, :, :, 1]
        b6 = dtcwt_output_h[0][:, :, 5, :, :, 1]

        c1 = torch.cat([a1, b1], dim=1)
        c2 = torch.cat([a2, b2], dim=1)
        c3 = torch.cat([a3, b3], dim=1)
        c4 = torch.cat([a4, b4], dim=1)
        c5 = torch.cat([a5, b5], dim=1)
        c6 = torch.cat([a6, b6], dim=1)

        new_y_outputl = torch.zeros_like(dtcwt_output_l)
        new_y_outputh = torch.zeros_like(dtcwt_output_h[0])

        # Process low-frequency and high-frequency components separately
        low_freq_processed = self.low_freq_conv(low_freq)
        low_freq_processed = self.igdn_low(low_freq_processed)

        c1_processed = self.conv1_new(c1)
        c1_processed = self.gdn1(c1_processed)
        c2_processed = self.conv2_new(c2)
        c2_processed = self.gdn2(c2_processed)
        c3_processed = self.conv3(c3)
        c3_processed = self.gdn3(c3_processed)
        c4_processed = self.conv4(c4)
        c4_processed = self.gdn4(c4_processed)
        c5_processed = self.conv5(c5)
        c5_processed = self.gdn5(c5_processed)
        c6_processed = self.conv6(c6)
        c6_processed = self.gdn6(c6_processed)

        a1_processed, b1_processed = torch.chunk(c1_processed, 2, dim=1)
        a2_processed, b2_processed = torch.chunk(c2_processed, 2, dim=1)
        a3_processed, b3_processed = torch.chunk(c3_processed, 2, dim=1)
        a4_processed, b4_processed = torch.chunk(c4_processed, 2, dim=1)
        a5_processed, b5_processed = torch.chunk(c5_processed, 2, dim=1)
        a6_processed, b6_processed = torch.chunk(c6_processed, 2, dim=1)

        new_y_outputh[:, :, 0, :, :, 0] = a1_processed
        new_y_outputh[:, :, 1, :, :, 0] = a2_processed
        new_y_outputh[:, :, 2, :, :, 0] = a3_processed
        new_y_outputh[:, :, 3, :, :, 0] = a4_processed
        new_y_outputh[:, :, 4, :, :, 0] = a5_processed
        new_y_outputh[:, :, 5, :, :, 0] = a6_processed
        new_y_outputh[:, :, 0, :, :, 1] = b1_processed
        new_y_outputh[:, :, 1, :, :, 1] = b2_processed
        new_y_outputh[:, :, 2, :, :, 1] = b3_processed
        new_y_outputh[:, :, 3, :, :, 1] = b4_processed
        new_y_outputh[:, :, 4, :, :, 1] = b5_processed
        new_y_outputh[:, :, 5, :, :, 1] = b6_processed

        new_y_outputh_list = []
        new_y_outputh_list.append(new_y_outputh)
        new_y_outputl = low_freq_processed

        # IDWT
        output = self.idtcwt((new_y_outputl, new_y_outputh_list))

        identity = self.upsample(x)
        output += identity
        return output



class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=True, groups=dim)

    def forward(self, x):
        x = rearrange(x, 'b h w c -> b c h w')
        x = self.dwconv(x)
        x = rearrange(x, 'b c h w -> b h w c')

        return x


class ConvolutionalGLU(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = hidden_features // 2
        self.fc1 = nn.Linear(in_features, hidden_features * 2)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()

        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x, v = self.fc1(x).chunk(2, dim=-1)
        x = self.act(self.dwconv(x)) * v
        x = self.fc2(x)
        return x


class Scale(nn.Module):
    def __init__(self, dim, init_value=1.0, trainable=True):
        super().__init__()
        self.scale = nn.Parameter(init_value * torch.ones(dim), requires_grad=trainable)

    def forward(self, x):
        return x * self.scale


class SpatialAttentionModule(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttentionModule, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class ConvWithDW(nn.Module):
    def __init__(self, input_dim=320, output_dim=320):
        super(ConvWithDW, self).__init__()
        self.in_trans = nn.Conv2d(input_dim, output_dim, kernel_size=1, padding=0, stride=1, bias=True)
        self.act1 = nn.GELU()
        self.dw_conv = nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1, stride=1, groups=output_dim,
                                 bias=True)
        self.act2 = nn.GELU()
        self.out_trans = nn.Conv2d(output_dim, output_dim, kernel_size=1, padding=0, stride=1, bias=True)

    def forward(self, x):
        x = self.in_trans(x)
        x = self.act1(x)
        x = self.dw_conv(x)
        x = self.act2(x)
        x = self.out_trans(x)
        return x


class DenseBlock(nn.Module):
    def __init__(self, dim=320):
        super(DenseBlock, self).__init__()
        self.layer_num = 3
        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.GELU(),
                ConvWithDW(dim, dim),
            ) for i in range(self.layer_num)
        ])
        self.proj = nn.Conv2d(dim * (self.layer_num + 1), dim, kernel_size=1, padding=0, stride=1, bias=True)

    def forward(self, x):
        outputs = [x]
        for i in range(self.layer_num):
            outputs.append(self.conv_layers[i](outputs[-1]))
        x = self.proj(torch.cat(outputs, dim=1))
        return x


class MultiScaleAggregation(nn.Module):
    def __init__(self, dim):
        super(MultiScaleAggregation, self).__init__()
        self.s = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, bias=True)
        self.spatial_atte = SpatialAttentionModule()
        self.dense = DenseBlock(dim)

    def forward(self, x):
        x = rearrange(x, 'b h w c -> b c h w')
        s = self.s(x)
        s_out = self.dense(s)
        x = s_out * self.spatial_atte(s_out)
        x = rearrange(x, 'b c h w -> b h w c')
        return x


class MutiScaleDictionaryCrossAttentionGLU(nn.Module):
    def __init__(self, input_dim, output_dim, mlp_rate=4, head_num=20, qkv_bias=True):
        super().__init__()

        dict_dim = 32 * head_num
        self.head_num = head_num

        self.scale = nn.Parameter(torch.ones(head_num, 1, 1))
        self.x_trans = nn.Linear(input_dim, dict_dim, bias=qkv_bias)

        self.ln_scale = nn.LayerNorm(dict_dim)
        self.msa = MultiScaleAggregation(dict_dim)

        self.lnx = nn.LayerNorm(dict_dim)
        self.q_trans = nn.Linear(dict_dim, dict_dim, bias=qkv_bias)
        self.dict_ln = nn.LayerNorm(dict_dim)
        self.k = nn.Linear(dict_dim, dict_dim, bias=qkv_bias)

        self.linear = nn.Linear(dict_dim, dict_dim, bias=qkv_bias)
        self.ln_mlp = nn.LayerNorm(dict_dim)

        self.mlp = ConvolutionalGLU(dict_dim, mlp_rate * dict_dim)
        self.output_trans = nn.Sequential(nn.Linear(dict_dim, output_dim))
        self.softmax = torch.nn.Softmax(dim=-1)

        self.res_scale_1 = Scale(dict_dim, init_value=1.0)
        self.res_scale_2 = Scale(dict_dim, init_value=1.0)
        self.res_scale_3 = Scale(dict_dim, init_value=1.0)

    def forward(self, x, dt):
        # 获取输入张量的维度
        B, C, H, W = x.size()
        # 将输入张量的维度从b c h w转换为b h w c
        x = rearrange(x, 'b c h w -> b h w c')
        # 对输入张量进行变换
        x = self.x_trans(x)

        # 对变换后的张量进行多头自注意力机制
        x = self.msa(self.ln_scale(x)) + self.res_scale_1(x)

        # 保存变换后的张量
        shortcut = x
        # 对变换后的张量进行线性变换
        x = self.lnx(x)
        # 对线性变换后的张量进行变换
        x = self.q_trans(x)
        # 将变换后的张量的维度从b h w c转换为b c h w
        x = rearrange(x, 'b h w c -> b c h w')

        # 将变换后的张量进行多头自注意力机制
        q = rearrange(x, 'b (e c) h w -> b e (h w) c', e=self.head_num)
        # 对输入的dt进行变换
        dt = self.dict_ln(dt)
        # 对变换后的dt进行线性变换
        k = self.k(dt)
        # 将变换后的k的维度从b n (e c)转换为b e n c
        k = rearrange(k, 'b n (e c) -> b e n c', e=self.head_num)
        # 将变换后的dt的维度从b n (e c)转换为b e n c
        dt = rearrange(dt, 'b n (e c) -> b e n c', e=self.head_num)
        # 将self.scale移动到q所在的设备
        self.scale = self.scale.to(q.device)
        # 计算q和k的相似度
        sim = torch.einsum('benc,bedc->bend', q, k)
        # 对相似度进行缩放
        sim = sim * self.scale
        # 计算相似度的概率分布
        probs = self.softmax(sim)
        # 计算输出
        output = torch.einsum('bend,bedc->benc', probs, dt)
        # 将输出的维度从b e (h w) c转换为b h w (e c)
        output = rearrange(output, 'b e (h w) c -> b h w (e c) ', h=H, w=W)

        # 对输出进行线性变换
        output = self.linear(output) + self.res_scale_2(shortcut)

        # 对线性变换后的输出进行多层感知机
        output = self.mlp(self.ln_mlp(output)) + self.res_scale_3(output)

        # 对多层感知机后的输出进行变换
        output = self.output_trans(output)
        # 将变换后的输出的维度从b h w c转换为b c h w
        output = rearrange(output, 'b h w c -> b c h w', )
        # 返回输出
        return output


class RSWeLIC(CompressionModel):
    def __init__(self, head_dim=[8, 16, 32, 32, 16, 8], drop_path_rate=0, N=192, M=320, num_slices=5,
                 max_support_slices=5,groups=None,groupss=None, **kwargs):
        super().__init__()

        if groups is None:
            groups = [0, 16, 16, 32, 64, 192]
        self.groups = list(groups)
        assert sum(self.groups) == M

        if groupss is None:
            groupss = [0, 3*16, 3*16, 3*32, 3*64, 3*192]
        self.groupss = list(groupss)
        assert sum(self.groupss) == 3*M

        self.head_dim = head_dim
        self.window_size = 8
        self.num_slices = num_slices
        self.max_support_slices = max_support_slices
        dim = N
        self.M = M
        begin = 0
        input_image_channel = 4
        output_image_channel = 4
        feature_dim = [96, 144, 256]

        # block_num = [0, 0, 4]
        block_num = [1, 2, 12]

        dict_num = 128
        dict_head_num = 20
        dict_dim = 32 * dict_head_num
        # 定义一个可学习的参数dt，其形状为[dict_num, dict_dim]，并设置requires_grad=True，表示需要计算梯度
        self.dt = nn.Parameter(torch.randn([dict_num, dict_dim]), requires_grad=True)
        self.dt_1 = nn.Parameter(torch.randn([dict_num, dict_dim]), requires_grad=True)
        prior_dim = M
        mlp_rate = 4
        qkv_bias = True
        self.dt_cross_attention = nn.ModuleList(
            MutiScaleDictionaryCrossAttentionGLU(input_dim=M * 2 + sum(self.groups[:i+1]), output_dim=M,
                                                 head_num=dict_head_num, mlp_rate=mlp_rate, qkv_bias=qkv_bias) for i in
            range(num_slices))
        self.dt_cross_attention_imag = nn.ModuleList(
            MutiScaleDictionaryCrossAttentionGLU(input_dim=M * 3 + sum(self.groupss[:i+1]), output_dim=M,
                                                 head_num=dict_head_num, mlp_rate=mlp_rate, qkv_bias=qkv_bias) for i in
            range(num_slices))


        self.g_a = nn.Sequential(
            ResidualBottleneckBlockWithStride_no(input_image_channel, feature_dim[0]),
            ConvNeXtV2Block(feature_dim[0], feature_dim[0]),
            ConvNeXtV2Block(feature_dim[0], feature_dim[0]),


            ResidualBottleneckBlockWithStride(feature_dim[0], feature_dim[1]),
            ConvNeXtV2Block(feature_dim[1], feature_dim[1]),
            ConvNeXtV2Block(feature_dim[1], feature_dim[1]),


            ResidualBottleneckBlockWithStride(feature_dim[1], feature_dim[2]),
            ConvNeXtV2Block(feature_dim[2], feature_dim[2]),
            ConvNeXtV2Block(feature_dim[2], feature_dim[2]),


            conv(feature_dim[2], M, kernel_size=5, stride=2),  # 亚像素卷积，上采样因子2（输出通道3对应RGB）
        )
        self.g_s = nn.Sequential(
            deconv(M, feature_dim[2], kernel_size=5, stride=1),

            ConvNeXtV2Block(feature_dim[2], feature_dim[2]),
            ConvNeXtV2Block(feature_dim[2], feature_dim[2]),

            ResidualBottleneckBlockWithUpsample(feature_dim[2], feature_dim[1]),

            ConvNeXtV2Block(feature_dim[1], feature_dim[1]),
            ConvNeXtV2Block(feature_dim[1], feature_dim[1]),

            ResidualBottleneckBlockWithUpsample(feature_dim[1], feature_dim[0]),

            ConvNeXtV2Block(feature_dim[0], feature_dim[0]),
            ConvNeXtV2Block(feature_dim[0], feature_dim[0]),

            ResidualBottleneckBlockWithUpsample(feature_dim[0], output_image_channel)  # 亚像素卷积，上采样因子2（输出通道3对应RGB）
        )

        # 超先验编码器 h_a：处理潜在表示生成超先验信息
        self.h_a = nn.Sequential(
            ResidualBottleneckBlockWithStride_no(4 * M, N,2),
            ResidualBlock(dim, dim),
            ResidualBlock(dim, dim),
            ResidualBlock(dim, dim),
            conv3x3(N, 192, stride=2)  # 3x3卷积下采样（输出通道192）
        )

        self.h_mean_s = nn.Sequential(
            deconv(192, N, kernel_size=3, stride=2),
            ResidualBlock(dim, dim),
            ResidualBlock(dim, dim),
            ResidualBlock(dim, dim),
            ResidualBottleneckBlockWithUpsample_no(N, M,2)
        )
        self.h_scale_s = nn.Sequential(
            deconv(192, N, kernel_size=3, stride=2),
            ResidualBlock(dim, dim),
            ResidualBlock(dim, dim),
            ResidualBlock(dim, dim),
            ResidualBottleneckBlockWithUpsample_no(N, M,2)
        )

        self.cc_mean_transforms = nn.ModuleList(
            nn.Sequential(
                conv(320 * 2 + sum(self.groups[:i+1]) + prior_dim, 224, stride=1, kernel_size=3),
                nn.GELU(),
                conv(224, 128, stride=1, kernel_size=3),
                nn.GELU(),
                conv(128, self.groups[i+1], stride=1, kernel_size=3),
            ) for i in range(self.num_slices)
        )
        self.cc_scale_transforms = nn.ModuleList(
            nn.Sequential(
                conv(320 * 2 + sum(self.groups[:i+1]) + prior_dim, 224, stride=1, kernel_size=3),
                nn.GELU(),
                conv(224, 128, stride=1, kernel_size=3),
                nn.GELU(),
                conv(128, self.groups[i+1], stride=1, kernel_size=3),
            ) for i in range(self.num_slices)
        )

        self.lrp_transforms = nn.ModuleList(
            nn.Sequential(
                conv(320 * 2 + sum(self.groups[:i+2]) + prior_dim, 224, stride=1, kernel_size=3),
                nn.GELU(),
                conv(224, 128, stride=1, kernel_size=3),
                nn.GELU(),
                conv(128, self.groups[i+1], stride=1, kernel_size=3),
            ) for i in range(self.num_slices)
        )

        self.cc_mean_transforms_imag = nn.ModuleList(
            nn.Sequential(
                conv(320 * 3 + 3*sum(self.groups[:i+1]) + prior_dim, 224, stride=1, kernel_size=3),
                nn.GELU(),
                conv(224, 128, stride=1, kernel_size=3),
                nn.GELU(),
                conv(128, 3*self.groups[i+1], stride=1, kernel_size=3),
            ) for i in range(self.num_slices)
        )
        self.cc_scale_transforms_imag = nn.ModuleList(
            nn.Sequential(
                conv(320 * 3 + 3*sum(self.groups[:i+1]) + prior_dim, 224, stride=1, kernel_size=3),
                nn.GELU(),
                conv(224, 128, stride=1, kernel_size=3),
                nn.GELU(),
                conv(128, 3*self.groups[i+1], stride=1, kernel_size=3),
            ) for i in range(self.num_slices)
        )

        self.lrp_transforms_imag = nn.ModuleList(
            nn.Sequential(
                conv(320 * 3 + 3*sum(self.groups[:i+2]) + prior_dim, 224, stride=1, kernel_size=3),
                nn.GELU(),
                conv(224, 128, stride=1, kernel_size=3),
                nn.GELU(),
                conv(128, 3*self.groups[i+1], stride=1, kernel_size=3),
            ) for i in range(self.num_slices)
        )

        self.entropy_bottleneck = EntropyBottleneck(192)

        self.gaussian_conditional_real = GaussianConditional(None)  # 实部高斯条件熵模型
        self.gaussian_conditional_imag = GaussianConditional(None)  # 虚部高斯条件熵模型

        self.dwt = DWT_2D(wave='haar')  # 二维离散小波变换层（如Haar小波）
        self.idwt = IDWT_2D(wave='haar')  # 二维逆小波变换层

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated_real = self.gaussian_conditional_real.update_scale_table(scale_table, force=force)
        updated_imag = self.gaussian_conditional_imag.update_scale_table(scale_table, force=force)
        updated_real |= super().update(force=force)
        updated_imag |= super().update(force=force)
        return updated_real, updated_imag

    def forward(self, x):
        b = x.size(0)
        # 将self.dt重复b次，生成一个新的张量dt
        dt = self.dt.repeat([b, 1, 1])
        dt_1 = self.dt_1.repeat([b, 1, 1])
        y = self.g_a(x)
        y_shape = y.shape[2:]

        y_output = self.dwt(y)

        # 分离低频和高频分量：低频为前320通道，高频为剩余通道
        low_freq = y_output[:, :320, :, :]
        high_freq = y_output[:, 320:, :, :]

        # 将低频和高频拼接作为超先验编码器的输入
        y_input = torch.cat([low_freq, high_freq], dim=1)

        z = self.h_a(y_input)
        _, z_likelihoods = self.entropy_bottleneck(z)
        z_offset = self.entropy_bottleneck._get_medians()
        z_tmp = z - z_offset
        z_hat = ste_round(z_tmp) + z_offset

        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)



        # 将低频分量分片处理（分片数量由num_slices定义）
        y_real_slices = torch.split(low_freq, self.groups[1:], 1)
        y_real_hat_slices = []  # 存储量化后的低频分片
        y_real_likelihood = []  # 存储低频分片的概率
        mu_real_list = []  # 存储低频分片的均值参数
        scale_real_list = []  # 存储低频分片的尺度参数

        # 高频分量分片处理
        y_imag_slices = torch.split(high_freq, self.groupss[1:], 1)
        y_imag_hat_slices = []
        y_imag_likelihood = []
        mu_imag_list = []
        scale_imag_list = []

        # 处理每个低频分片
        for slice_index, y_slice in enumerate(y_real_slices):
            # 获取当前分片的上下文支持分片（最多max_support_slices个）
            support_slices = (y_real_hat_slices if self.max_support_slices < 0
                              else y_real_hat_slices[:self.max_support_slices])
            query = torch.cat([latent_scales] + [latent_means] + support_slices, dim=1)
            dict_info = self.dt_cross_attention[slice_index](query, dt)
            support = torch.cat([query] + [dict_info], dim=1)

            mu = self.cc_mean_transforms[slice_index](supp=ort)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]
            mu_real_list.append(mu)

            scale = self.cc_scale_transforms[slice_index](support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]
            scale_real_list.append(scale)

            _, y_slice_likelihood = self.gaussian_conditional_real(y_slice, scale, mu)
            y_real_likelihood.append(y_slice_likelihood)
            # 量化操作：y_hat = round(y - mu) + mu
            y_hat_slice = ste_round(y_slice - mu) + mu
            # 潜在残差预测（LRP）模块优化量化结果
            lrp_support = torch.cat([support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)  # 使用tanh限制残差范围
            y_hat_slice += lrp
            y_real_hat_slices.append(y_hat_slice)

        y_real_hat = torch.cat(y_real_hat_slices, dim=1)
        means_real = torch.cat(mu_real_list, dim=1)
        scales_real = torch.cat(scale_real_list, dim=1)
        y_real_likelihoods = torch.cat(y_real_likelihood, dim=1)

        # 处理每个低频分片
        for slice_index, y_slice in enumerate(y_imag_slices):
            # 获取当前分片的上下文支持分片（最多max_support_slices个）
            support_slices = (y_imag_hat_slices if self.max_support_slices < 0
                              else y_imag_hat_slices[:self.max_support_slices])
            query = torch.cat([latent_scales] + [latent_means] + [y_real_hat] + support_slices, dim=1)
            dict_info = self.dt_cross_attention_imag[slice_index](query, dt_1)
            support = torch.cat([query] + [dict_info], dim=1)

            mu = self.cc_mean_transforms_imag[slice_index](support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]
            mu_imag_list.append(mu)

            scale = self.cc_scale_transforms_imag[slice_index](support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]
            scale_imag_list.append(scale)

            _, y_slice_likelihood = self.gaussian_conditional_imag(y_slice, scale, mu)
            y_imag_likelihood.append(y_slice_likelihood)
            # 量化操作：y_hat = round(y - mu) + mu
            y_hat_slice = ste_round(y_slice - mu) + mu
            # 潜在残差预测（LRP）模块优化量化结果
            lrp_support = torch.cat([support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms_imag[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)  # 使用tanh限制残差范围
            y_hat_slice += lrp
            y_imag_hat_slices.append(y_hat_slice)

        y_imag_hat = torch.cat(y_imag_hat_slices, dim=1)
        means_imag = torch.cat(mu_imag_list, dim=1)
        scales_imag = torch.cat(scale_imag_list, dim=1)
        y_imag_likelihoods = torch.cat(y_imag_likelihood, dim=1)

        # 拼接低频和高频结果，进行逆小波变换（IDWT）重建潜在表示
        dwt_processed = torch.cat([y_real_hat, y_imag_hat], dim=1)
        y_hat = self.idwt(dwt_processed)

        x_hat = self.g_s(y_hat)
        return {
            "x_hat": x_hat,
            "likelihoods": {
                "y_real": y_real_likelihoods,
                "y_imag": y_imag_likelihoods,
                "z": z_likelihoods
            },
            "para": {"means": means_real, "scales": scales_real, "y": y}
        }



    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = state_dict["g_a.0.weight"].size(0)
        M = state_dict["g_a.6.weight"].size(0)
        net = cls(N, M)
        net.load_state_dict(state_dict)
        return net

    def compress(self, x):
        b = x.size(0)
        dt = self.dt.repeat([b, 1, 1])
        dt_1 = self.dt_1.repeat([b, 1, 1])
        y = self.g_a(x)
        y_shape = y.shape[2:]

        y_output = self.dwt(y)

        low_freq = y_output[:, :320, :, :]
        high_freq = y_output[:, 320:, :, :]
        y_input = torch.cat([low_freq, high_freq], dim=1)

        z = self.h_a(y_input)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        # 将低频分量分片处理（分片数量由num_slices定义）
        y_real_slices = torch.split(low_freq, self.groups[1:], 1)
        y_real_hat_slices = []  # 存储量化后的低频分片
        y_real_likelihood = []  # 存储低频分片的概率
        mu_real_list = []  # 存储低频分片的均值参数
        scale_real_list = []  # 存储低频分片的尺度参数


        cdf_real = self.gaussian_conditional_real.quantized_cdf.tolist()
        cdf_real_lengths = self.gaussian_conditional_real.cdf_length.reshape(-1).int().tolist()
        offsets_real = self.gaussian_conditional_real.offset.reshape(-1).int().tolist()

        encoder_real = BufferedRansEncoder()
        symbols_real_list = []
        indexes_real_list = []
        y_real_strings = []

        # 处理每个低频分片
        for slice_index, y_slice in enumerate(y_real_slices):
            # 获取当前分片的上下文支持分片（最多max_support_slices个）
            support_slices = (y_real_hat_slices if self.max_support_slices < 0
                              else y_real_hat_slices[:self.max_support_slices])
            query = torch.cat([latent_scales] + [latent_means] + support_slices, dim=1)
            dict_info = self.dt_cross_attention[slice_index](query, dt)
            support = torch.cat([query] + [dict_info], dim=1)

            mu = self.cc_mean_transforms[slice_index](support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]
            scale = self.cc_scale_transforms[slice_index](support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            index = self.gaussian_conditional_real.build_indexes(scale)
            y_q_slice = self.gaussian_conditional_real.quantize(y_slice, "symbols", mu)
            y_hat_slice = y_q_slice + mu

            symbols_real_list.extend(y_q_slice.reshape(-1).tolist())
            indexes_real_list.extend(index.reshape(-1).tolist())

            lrp_support = torch.cat([support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)  # 使用tanh限制残差范围
            y_hat_slice += lrp
            y_real_hat_slices.append(y_hat_slice)
            scale_real_list.append(scale)
            mu_real_list.append(mu)

        encoder_real.encode_with_indexes(symbols_real_list, indexes_real_list, cdf_real, cdf_real_lengths, offsets_real)
        y_real_string = encoder_real.flush()
        y_real_strings.append(y_real_string)

        # 高频分量分片处理
        y_imag_slices = torch.split(high_freq, self.groupss[1:], 1)
        y_imag_hat_slices = []
        y_imag_likelihood = []
        mu_imag_list = []
        scale_imag_list = []

        y_real_hat = torch.cat(y_real_hat_slices, dim=1)
        encoder_imag = BufferedRansEncoder()
        symbols_imag_list = []
        indexes_imag_list = []
        y_imag_strings = []

        cdf_imag = self.gaussian_conditional_imag.quantized_cdf.tolist()
        cdf_imag_lengths = self.gaussian_conditional_imag.cdf_length.reshape(-1).int().tolist()
        offsets_imag = self.gaussian_conditional_imag.offset.reshape(-1).int().tolist()

        for slice_index, y_slice in enumerate(y_imag_slices):
            # 获取当前分片的上下文支持分片（最多max_support_slices个）
            support_slices = (y_imag_hat_slices if self.max_support_slices < 0
                              else y_imag_hat_slices[:self.max_support_slices])
            query = torch.cat([latent_scales] + [latent_means]+ [y_real_hat] + support_slices, dim=1)
            dict_info = self.dt_cross_attention_imag[slice_index](query, dt_1)
            support = torch.cat([query] + [dict_info], dim=1)

            mu = self.cc_mean_transforms_imag[slice_index](support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]
            scale = self.cc_scale_transforms_imag[slice_index](support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            index = self.gaussian_conditional_imag.build_indexes(scale)
            y_q_slice = self.gaussian_conditional_imag.quantize(y_slice, "symbols", mu)
            y_hat_slice = y_q_slice + mu

            symbols_imag_list.extend(y_q_slice.reshape(-1).tolist())
            indexes_imag_list.extend(index.reshape(-1).tolist())

            lrp_support = torch.cat([support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms_imag[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)  # 使用tanh限制残差范围
            y_hat_slice += lrp
            y_imag_hat_slices.append(y_hat_slice)
            scale_imag_list.append(scale)
            mu_imag_list.append(mu)

        encoder_imag.encode_with_indexes(symbols_imag_list, indexes_imag_list, cdf_imag, cdf_imag_lengths, offsets_imag)
        y_imag_string = encoder_imag.flush()
        y_imag_strings.append(y_imag_string)

        return {"strings": [y_real_strings, y_imag_strings, z_strings], "shape": z.size()[-2:]}

    def _likelihood(self, inputs, scales, means=None):
        half = float(0.5)
        if means is not None:
            values = inputs - means
        else:
            values = inputs

        scales = torch.max(scales, torch.tensor(0.11))
        values = torch.abs(values)
        upper = self._standardized_cumulative((half - values) / scales)
        lower = self._standardized_cumulative((-half - values) / scales)
        likelihood = upper - lower
        return likelihood

    def _standardized_cumulative(self, inputs):
        half = float(0.5)
        const = float(-(2 ** -0.5))
        # Using the complementary error function maximizes numerical precision.
        return half * torch.erfc(const * inputs)

    def decompress(self, strings, shape):

        z_hat = self.entropy_bottleneck.decompress(strings[2], shape)
        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        b = z_hat.size(0)
        dt = self.dt.repeat([b, 1, 1])
        dt_1 = self.dt_1.repeat([b, 1, 1])
        y_shape = [z_hat.shape[2] * 4, z_hat.shape[3] * 4]

        y_real_string = strings[0][0]
        y_imag_string = strings[1][0]

        y_real_hat_slices = []
        cdf_real = self.gaussian_conditional_real.quantized_cdf.tolist()
        cdf_real_lengths = self.gaussian_conditional_real.cdf_length.reshape(-1).int().tolist()
        offsets_real = self.gaussian_conditional_real.offset.reshape(-1).int().tolist()

        decoder_real = RansDecoder()
        decoder_real.set_stream(y_real_string)

        for slice_index in range(self.num_slices):
            support_slices = (y_real_hat_slices if self.max_support_slices < 0 else y_real_hat_slices[:self.max_support_slices])
            query = torch.cat([latent_scales] + [latent_means] + support_slices, dim=1)
            dict_info = self.dt_cross_attention[slice_index](query, dt)
            support = torch.cat([query] + [dict_info], dim=1)

            mu = self.cc_mean_transforms[slice_index](support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale = self.cc_scale_transforms[slice_index](support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            index = self.gaussian_conditional_real.build_indexes(scale)

            rv = decoder_real.decode_stream(index.reshape(-1).tolist(), cdf_real, cdf_real_lengths, offsets_real)
            rv = torch.Tensor(rv).reshape(1, -1, y_shape[0], y_shape[1])
            y_real_hat_slice = self.gaussian_conditional_real.dequantize(rv, mu)

            lrp_support = torch.cat([support, y_real_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_real_hat_slice += lrp

            y_real_hat_slices.append(y_real_hat_slice)

        y_real_hat = torch.cat(y_real_hat_slices, dim=1)

        y_imag_hat_slices = []
        cdf_imag = self.gaussian_conditional_imag.quantized_cdf.tolist()
        cdf_imag_lengths = self.gaussian_conditional_imag.cdf_length.reshape(-1).int().tolist()
        offsets_imag = self.gaussian_conditional_imag.offset.reshape(-1).int().tolist()

        decoder_imag = RansDecoder()
        decoder_imag.set_stream(y_imag_string)


        for slice_index in range(self.num_slices):
            support_slices = (y_imag_hat_slices if self.max_support_slices < 0 else y_imag_hat_slices[:self.max_support_slices])
            query = torch.cat([latent_scales] + [latent_means] + [y_real_hat]+ support_slices, dim=1)
            dict_info = self.dt_cross_attention_imag[slice_index](query, dt_1)
            support = torch.cat([query] + [dict_info], dim=1)

            mu = self.cc_mean_transforms_imag[slice_index](support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale = self.cc_scale_transforms_imag[slice_index](support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            index = self.gaussian_conditional_imag.build_indexes(scale)

            rv = decoder_imag.decode_stream(index.reshape(-1).tolist(), cdf_imag, cdf_imag_lengths, offsets_imag)
            rv = torch.Tensor(rv).reshape(1, -1, y_shape[0], y_shape[1])
            y_imag_hat_slice = self.gaussian_conditional_imag.dequantize(rv, mu)

            lrp_support = torch.cat([support, y_imag_hat_slice], dim=1)
            lrp = self.lrp_transforms_imag[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_imag_hat_slice += lrp

            y_imag_hat_slices.append(y_imag_hat_slice)

        y_imag_hat = torch.cat(y_imag_hat_slices, dim=1)

        # y_hat = torch.cat((y_real_hat, y_imag_hat), 1)

        dwt_processed = torch.cat([y_real_hat, y_imag_hat], dim=1)

        # IDWT
        y_hat = self.idwt(dwt_processed)

        x_hat = self.g_s(y_hat).clamp_(0, 1)

        return {"x_hat": x_hat}

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None
class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)
class NAFBlock(nn.Module):
    def __init__(self, dim, inter_dim=None):
        super().__init__()

        self.dim = inter_dim if inter_dim is not None else dim

        dw_channel = self.dim << 1
        ffn_channel = self.dim << 1

        self.dwconv = nn.Sequential(
            nn.Conv2d(self.dim, dw_channel, 1),
            nn.Conv2d(dw_channel, dw_channel, 3, 1, padding=1, groups=dw_channel)
        )
        self.sg = SimpleGate()
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channel // 2, dw_channel // 2, 1)
        )
        self.FFN = nn.Sequential(
            nn.Conv2d(self.dim, ffn_channel, 1),
            SimpleGate(),
            nn.Conv2d(ffn_channel >> 1, self.dim, 1)
        )

        self.norm1 = LayerNorm2d(self.dim)
        self.norm2 = LayerNorm2d(self.dim)
        self.conv1 = nn.Conv2d(dw_channel >> 1, self.dim, 1)

        self.beta = nn.Parameter(torch.zeros((1, self.dim, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, self.dim, 1, 1)), requires_grad=True)

        self.in_conv = conv(dim, inter_dim, kernel_size=1, stride=1) if inter_dim is not None else nn.Identity()
        self.out_conv = conv(inter_dim, dim, kernel_size=1, stride=1) if inter_dim is not None else nn.Identity()

    def forward(self, x):
        x = self.in_conv(x)
        identity = x
        x = self.norm1(x)

        x = self.dwconv(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv1(x)

        out = identity + x * self.beta
        identity = out

        out = self.norm2(out)
        out = self.FFN(out)

        out = identity + out * self.gamma

        out = self.out_conv(out)
        return out

class LayerNorm(nn.Module):
    """ LayerNorm with channels_first support """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        else:
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class GRN(nn.Module):
    """ Global Response Normalization """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x

class ConvNeXtV2Block(nn.Module):
    """ 替换ResidualBlock的ConvNeXtV2 Block """
    def __init__(self, in_ch, out_ch, drop_path=0.05):
        super().__init__()
        # 通道对齐
        if in_ch != out_ch:
            self.channel_align = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        else:
            self.channel_align = nn.Identity()

        # ConvNeXtV2 Block核心结构
        # 定义深度卷积层
        self.dwconv = nn.Conv2d(out_ch, out_ch, kernel_size=7, padding=3, groups=out_ch)
        # 定义层归一化层
        self.norm = LayerNorm(out_ch, eps=1e-6, data_format="channels_first")
        # 定义全连接层
        self.pwconv1 = nn.Linear(out_ch, 4 * out_ch)
        # 定义激活函数
        self.act = nn.GELU()
        # 定义GRN层
        self.grn = GRN(4 * out_ch)
        # 定义全连接层
        self.pwconv2 = nn.Linear(4 * out_ch, out_ch)
        # 定义DropPath层
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x):
        identity = x
        x = self.channel_align(x)  # 通道对齐
        x = self.dwconv(x)
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1)  # (N,C,H,W) -> (N,H,W,C)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # (N,H,W,C) -> (N,C,H,W)
        return identity + self.drop_path(x)


if __name__ == "__main__":
    print(torch.cuda.is_available())
    model = RSWeLIC()
    model = model.cuda()
    input = torch.Tensor(1, 4, 256, 256)
    output = model(input.cuda())
    print(output)