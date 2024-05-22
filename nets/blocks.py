from ast import main
from turtle import forward
from SimpleITK import Sigmoid
import numpy as np
import torch
import math
import torch.nn as nn
from typing import Union, Type, List, Tuple
import torch.nn.functional as F
from mamba_ssm import Mamba
from torch.cuda.amp import autocast
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd
from dynamic_network_architectures.building_blocks.helper import maybe_convert_scalar_to_list, get_matching_pool_op




class MambaLayer(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
            d_model=dim,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand,  # Block expansion factor
        )

    @autocast(enabled=False)
    def forward(self, x,dummy_tensor=None):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
        B, C = x.shape[:2]
        assert C == self.dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm)
        out = x_mamba.transpose(-1, -2).reshape(B, C, *img_dims)
        return out


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm3d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                # avg_pool = F.avg_pool3d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                avg_pool = F.avg_pool3d(x, kernel_size=(x.size(2), x.size(3), x.size(4)), stride=(x.size(2), x.size(3), x.size(4)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                # max_pool = F.max_pool3d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                max_pool = F.max_pool3d(x, kernel_size=(x.size(2), x.size(3), x.size(4)), stride=(x.size(2), x.size(3), x.size(4)))
                channel_att_raw = self.mlp( max_pool )
            # elif pool_type=='lp':
            #     lp_pool = F.lp_pool3d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
            #     channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_3d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).unsqueeze(4).expand_as(x)
        return x * scale

def logsumexp_3d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out) # broadcasting
        return x * scale

class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
    def forward(self, x,dummy_tensor=None):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out
    
class ConvDropoutNormReLU(nn.Module):
    def __init__(   self,
                    conv_op: Type[_ConvNd],
                    input_channels: int,
                    output_channels: int,
                    kernel_size: Union[int, List[int], Tuple[int, ...]],
                    stride: Union[int, List[int], Tuple[int, ...]],
                    conv_bias: bool = False,
                    norm_op: Union[None, Type[nn.Module]] = None,
                    norm_op_kwargs: dict = None,
                    dropout_op: Union[None, Type[_DropoutNd]] = None,
                    dropout_op_kwargs: dict = None,
                    nonlin: Union[None, Type[torch.nn.Module]] = None,
                    nonlin_kwargs: dict = None,
                    nonlin_first: bool = False
                    ):
        super(ConvDropoutNormReLU, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        stride = maybe_convert_scalar_to_list(conv_op, stride)
        self.stride = stride

        kernel_size = maybe_convert_scalar_to_list(conv_op, kernel_size)
        if norm_op_kwargs is None:
            norm_op_kwargs = {}
        if nonlin_kwargs is None:
            nonlin_kwargs = {}

        ops = []

        self.conv = conv_op(
            input_channels,
            output_channels,
            kernel_size,
            stride,
            padding=[(i - 1) // 2 for i in kernel_size],
            dilation=1,
            bias=conv_bias,
        )
        ops.append(self.conv)

        if dropout_op is not None:
            self.dropout = dropout_op(**dropout_op_kwargs)
            ops.append(self.dropout)

        if norm_op is not None:
            self.norm = norm_op(output_channels, **norm_op_kwargs)
            ops.append(self.norm)

        if nonlin is not None:
            self.nonlin = nonlin(**nonlin_kwargs)
            ops.append(self.nonlin)

        if nonlin_first and (norm_op is not None and nonlin is not None):
            ops[-1], ops[-2] = ops[-2], ops[-1]

        self.all_modules = nn.Sequential(*ops)

    def forward(self, x):
        return self.all_modules(x)

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == len(self.stride), "just give the image size without color/feature channels or " \
                                                    "batch channel. Do not give input_size=(b, c, x, y(, z)). " \
                                                    "Give input_size=(x, y(, z))!"
        output_size = [i // j for i, j in zip(input_size, self.stride)]  # we always do same padding
        return np.prod([self.output_channels, *output_size], dtype=np.int64)



class MutiFuse(nn.Module):
    def __init__(self, 
                in_channels:int, 
                out_channels:int, 
                # exp_r:int=4, 
                kernel_size:int=3, 
                # do_res:int=True,
                norm_type:str = 'group',
                n_groups:int or None = None,
                dim = '3d',
                *args,
                **kwargs) -> None:
        super(MutiFuse,self).__init__(*args, **kwargs)
        
        assert dim in ['2d', '3d']
        self.dim = dim
        if self.dim == '2d':
            conv = nn.Conv2d
        elif self.dim == '3d':
            conv = nn.Conv3d
        
        kernel_size = 1
        self.cbam = CBAM(gate_channels=in_channels)
        self._conv1d = conv(in_channels=in_channels,
                            out_channels=in_channels,
                            kernel_size=kernel_size,
                            stride=1,
                            padding = kernel_size//2,
                            groups = in_channels if n_groups is None else n_groups,)
        self.conv1d = conv(in_channels=in_channels * 2,
                            out_channels=in_channels ,
                            kernel_size=kernel_size,
                            stride=1,
                            padding = 0,
                            groups = in_channels if n_groups is None else n_groups,)
        
        # self.conv1d = nn.Conv3d(
        #     in_channels=in_channels * 2,  # 假设这是输入特征图的通道数
        #     out_channels=in_channels,  # 保持输出通道数与输入相同
        #     kernel_size=1,  # 使用1x1的卷积核
        #     stride=1,  # 步长为1，保持特征图大小不变
        #     padding=0,  # 由于卷积核大小为1，所以不需要填充
        #     groups=1  # 通常1x1卷积不需要分组，除非有特殊需求
        #     )
        self.stack_conv_norm_relu = ConvDropoutNormReLU(conv_op=nn.Conv3d,input_channels=in_channels * 2,kernel_size=3,\
            output_channels=in_channels * 2,stride=1)
        self.cbam = CBAM(in_channels * 2 )
        
        if in_channels == out_channels:
            self.skip = nn.Identity()
        else:
            self.skip = conv(inchannels=in_channels * 2,out_channels=in_channels ,kernel_size=1,stride=1)
            
    def forward(self,input,skip):
        _x = self._conv1d(input)
        _s = self._conv1d(skip)
        _stage1 = torch.sigmoid(_x + _s)
        del _x, _s
        
        stage0 = torch.concat((input,skip),dim=1)
        x1 = self.stack_conv_norm_relu(stage0)
        x2 = self.cbam(stage0)
        x3 = self.skip(stage0)
        # 假设 x1, x2, x3 的形状相同
        temp =x1 * x2 * x3
        stage1 = self.conv1d(temp) 
        del x1, x2, x3

        return _stage1 * stage1
        
if __name__ == '__main__' :
    model = MutiFuse(in_channels=32,out_channels=32,dim='3d',n_groups=16).cuda()
    x = torch.zeros((1, 32, 64, 64, 64), requires_grad=False).cuda()
    y = model(x,x)
    print(y.shape)