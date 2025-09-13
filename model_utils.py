# model_utils.py
import streamlit as st
import torch
import torch.nn as nn
import os
from config import CLASSES # Ambil dari config.py
from collections.abc import Sequence
from monai.networks.nets import UNet


# --- Salin semua definisi kelas model (SingleConvolution, DoubleConvolution, ChannelGate, SpatialGate, CBAM, DownSampling, UpSampling, Attention_Unet) ke sini ---
# Contoh:

# ATTENTION GATE

# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

__all__ = ["AttentionUnet"]


class ConvBlock(nn.Module):

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Sequence[int] | int = 3,
        strides: int = 1,
        dropout=0.0,
    ):
        super().__init__()
        layers = [
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                strides=strides,
                padding=None,
                adn_ordering="NDA",
                act="relu",
                norm=Norm.BATCH,
                dropout=dropout,
            ),
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                strides=1,
                padding=None,
                adn_ordering="NDA",
                act="relu",
                norm=Norm.BATCH,
                dropout=dropout,
            ),
        ]
        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_c: torch.Tensor = self.conv(x)
        return x_c


class UpConv(nn.Module):

    def __init__(self, spatial_dims: int, in_channels: int, out_channels: int, kernel_size=3, strides=2, dropout=0.0):
        super().__init__()
        self.up = Convolution(
            spatial_dims,
            in_channels,
            out_channels,
            strides=strides,
            kernel_size=kernel_size,
            act="relu",
            adn_ordering="NDA",
            norm=Norm.BATCH,
            dropout=dropout,
            is_transposed=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_u: torch.Tensor = self.up(x)
        return x_u


class AttentionBlock(nn.Module):

    def __init__(self, spatial_dims: int, f_int: int, f_g: int, f_l: int, dropout=0.0):
        super().__init__()
        self.W_g = nn.Sequential(
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=f_g,
                out_channels=f_int,
                kernel_size=1,
                strides=1,
                padding=0,
                dropout=dropout,
                conv_only=True,
            ),
            Norm[Norm.BATCH, spatial_dims](f_int),
        )

        self.W_x = nn.Sequential(
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=f_l,
                out_channels=f_int,
                kernel_size=1,
                strides=1,
                padding=0,
                dropout=dropout,
                conv_only=True,
            ),
            Norm[Norm.BATCH, spatial_dims](f_int),
        )

        self.psi = nn.Sequential(
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=f_int,
                out_channels=1,
                kernel_size=1,
                strides=1,
                padding=0,
                dropout=dropout,
                conv_only=True,
            ),
            Norm[Norm.BATCH, spatial_dims](1),
            nn.Sigmoid(),
        )

        self.relu = nn.ReLU()

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi: torch.Tensor = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


class AttentionLayer(nn.Module):

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        submodule: nn.Module,
        up_kernel_size=3,
        strides=2,
        dropout=0.0,
    ):
        super().__init__()
        self.attention = AttentionBlock(
            spatial_dims=spatial_dims, f_g=in_channels, f_l=in_channels, f_int=in_channels // 2
        )
        self.upconv = UpConv(
            spatial_dims=spatial_dims,
            in_channels=out_channels,
            out_channels=in_channels,
            strides=strides,
            kernel_size=up_kernel_size,
        )
        self.merge = Convolution(
            spatial_dims=spatial_dims, in_channels=2 * in_channels, out_channels=in_channels, dropout=dropout
        )
        self.submodule = submodule

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fromlower = self.upconv(self.submodule(x))
        att = self.attention(g=fromlower, x=x)
        att_m: torch.Tensor = self.merge(torch.cat((att, fromlower), dim=1))
        return att_m


class AttentionUnet(nn.Module):
    """
    Attention Unet based on
    Otkay et al. "Attention U-Net: Learning Where to Look for the Pancreas"
    https://arxiv.org/abs/1804.03999

    Args:
        spatial_dims: number of spatial dimensions of the input image.
        in_channels: number of the input channel.
        out_channels: number of the output classes.
        channels (Sequence[int]): sequence of channels. Top block first. The length of `channels` should be no less than 2.
        strides (Sequence[int]): stride to use for convolutions.
        kernel_size: convolution kernel size.
        up_kernel_size: convolution kernel size for transposed convolution layers.
        dropout: dropout ratio. Defaults to no dropout.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        channels: Sequence[int],
        strides: Sequence[int],
        kernel_size: Sequence[int] | int = 3,
        up_kernel_size: Sequence[int] | int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dimensions = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = channels
        self.strides = strides
        self.kernel_size = kernel_size
        self.dropout = dropout

        head = ConvBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=channels[0],
            dropout=dropout,
            kernel_size=self.kernel_size,
        )
        reduce_channels = Convolution(
            spatial_dims=spatial_dims,
            in_channels=channels[0],
            out_channels=out_channels,
            kernel_size=1,
            strides=1,
            padding=0,
            conv_only=True,
        )
        self.up_kernel_size = up_kernel_size

        def _create_block(channels: Sequence[int], strides: Sequence[int]) -> nn.Module:
            if len(channels) > 2:
                subblock = _create_block(channels[1:], strides[1:])
                return AttentionLayer(
                    spatial_dims=spatial_dims,
                    in_channels=channels[0],
                    out_channels=channels[1],
                    submodule=nn.Sequential(
                        ConvBlock(
                            spatial_dims=spatial_dims,
                            in_channels=channels[0],
                            out_channels=channels[1],
                            strides=strides[0],
                            dropout=self.dropout,
                            kernel_size=self.kernel_size,
                        ),
                        subblock,
                    ),
                    up_kernel_size=self.up_kernel_size,
                    strides=strides[0],
                    dropout=dropout,
                )
            else:
                # the next layer is the bottom so stop recursion,
                # create the bottom layer as the subblock for this layer
                return self._get_bottom_layer(channels[0], channels[1], strides[0])

        encdec = _create_block(self.channels, self.strides)
        self.model = nn.Sequential(head, encdec, reduce_channels)

    def _get_bottom_layer(self, in_channels: int, out_channels: int, strides: int) -> nn.Module:
        return AttentionLayer(
            spatial_dims=self.dimensions,
            in_channels=in_channels,
            out_channels=out_channels,
            submodule=ConvBlock(
                spatial_dims=self.dimensions,
                in_channels=in_channels,
                out_channels=out_channels,
                strides=strides,
                dropout=self.dropout,
                kernel_size=self.kernel_size,
            ),
            up_kernel_size=self.up_kernel_size,
            strides=strides,
            dropout=self.dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_m: torch.Tensor = self.model(x)
        return x_m
    
# SE

class DoubleConvolution(nn.Module):
    """
    Auxiliary class to define a convolutional layer.
    Each convolution block: 3x3 convolution, batch normalization, ReLU activation.

    Args:
        nn.Module : receive the nn.Module properties
    """
    def __init__(self, in_channels : int, out_channels : int) -> None:
        """
        Args:
            in_channels (int): amount of input channels (16 or 32 or 64 or 128)
            out_channels (int): amount of output channels (16 or 32 or 64 or 128)
        """ 
        super(DoubleConvolution, self).__init__()
        
        self.doubleConv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size = 3, padding=1),
            nn.BatchNorm3d(out_channels), 
            nn.ReLU(inplace=True),
            
            nn.Conv3d(out_channels, out_channels, kernel_size = 3, padding=1),
            nn.BatchNorm3d(out_channels), 
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor

        Returns:
            torch.Tensor: output tensor
        """
        return self.doubleConv(x)

class SE(nn.Module):
    """
    Auxiliary class to define a squeeze and excitation layer.

    Args:
        nn.Module: receive the nn.Module properties
    """
    def __init__(self, in_channels : int) -> None:
        """
        Args:
            in_channels (int): amount of input channels (16 or 32 or 64 or 128)
        """        
        super(SE, self).__init__()
        
        self.squeeze = nn.AdaptiveAvgPool3d(1) # Global Average Pooling
        self.excitation = nn.Sequential(
            nn.Linear(in_channels, in_channels // 8), # Reduction ratio = 8
            nn.ReLU(inplace=True), # ReLU activation
            nn.Linear(in_channels // 8, in_channels), # Increase ratio = 8
            nn.Sigmoid() # Sigmoid activation
        )
        
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor

        Returns:
            torch.Tensor: output tensor
        """        
        batch_size, channels, _, _, _ = x.size()
        y = self.squeeze(x).view(batch_size, channels)
        y = self.excitation(y).view(batch_size, channels, 1, 1, 1)
        return x * y.expand_as(x)

class DownSampling(nn.Module):
    """
    Auxiliary class to define a downsampling layer.
    Each downsampling block: 2x2 max pooling, double convolution and squeeze and excitation.
    input X output: [1, 16, 128, 128, 128] ->  [1, 32, 64, 64, 64] 
                    [1, 32, 64, 64, 64]    ->  [1, 64, 32, 32, 32]
                    [1, 64, 32, 32, 32]    ->  [1, 128, 16, 16, 16]

    Args:
        nn.Module: receive the nn.Module properties
    """
    def __init__(self, in_channels : int, out_channels : int) -> None:
        """
        Args:
            in_channels (int): amount of input channels (16 or 32 or 64)
            out_channels (int): amount of output channels (32 or 64 or 128)
        """        
        super(DownSampling, self).__init__()
        
        self.maxpool = nn.MaxPool3d(2)
        self.conv = DoubleConvolution(in_channels, out_channels)
        self.attention = SE(out_channels)
        
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): _description_

        Returns:
            torch.Tensor: _description_
        """
        out = self.maxpool(x) # 2x2 max pooling -> 1/2 the size but same amount of channels
        out = self.conv(out) # double convolution -> same size but double the amount of channels
        out = self.attention(out) # squeeze and excitation
        return out

class UpSampling(nn.Module):
    """
    Auxiliary class to define a upsampling layer.
    Each upsampling block: 2x2 upsampling, concatenation with skip connection, double convolution.
    input X output: [1, 128, 16, 16, 16] ->  [1, 64, 32, 32, 32]
                    [1, 64, 32, 32, 32]    ->  [1, 32, 64, 64, 64]
                    [1, 32, 64, 64, 64]    ->  [1, 16, 128, 128, 128]
                    
    Args:
        nn.Module: receive the nn.Module properties
    """
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = False) -> None:
        """
        Args:
            in_channels (int): amount of input channels (128 or 64 or 32)
            out_channels (int): amount of output channels (64 or 32 or 16)
        """
        super(UpSampling, self).__init__()
        
        self.up = nn.ConvTranspose3d(in_channels, in_channels, kernel_size=2, stride=2)
        self.conv = DoubleConvolution(int(in_channels + out_channels), out_channels)
        
    def forward(self, x : torch.Tensor, skip_connection : torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): the input tensor
            skip_connection (torch.Tensor): the skip connection from the downsampling path

        Returns:
            torch.Tensor: the output tensor
        """
        x = self.up(x) # 2x2 upsampling -> double the size but same amount of channels
        x = torch.cat([skip_connection, x], dim=1) # concatenation with skip connection
        out = self.conv(x) # double convolution -> same size but half the amount of channels
        return out

class SqueezeAndExcitation3DUnet(nn.Module):
        def __init__(self, in_channels=1, out_channels=1, channels=(32, 64, 128, 128)) -> None:
            super(SqueezeAndExcitation3DUnet, self).__init__()
            layers = channels
            
            self.input = nn.Sequential(DoubleConvolution(in_channels, layers[0]), SE(layers[0])) # tranform the input to 16 channels and apply squeeze and excitation
            # encoding path
            self.down1 = DownSampling(layers[0], layers[1]) 
            self.down2 = DownSampling(layers[1], layers[2]) 
            self.down3 = DownSampling(layers[2], layers[3])
            # decoding path
            self.up1 = UpSampling(layers[3], layers[2])
            self.up2 = UpSampling(layers[2], layers[1])
            self.up3 = UpSampling(layers[1], layers[0])
            self.output = nn.Sequential(nn.Conv3d(layers[0], out_channels, kernel_size=1)) # transform the output
        
        def forward(self, x : torch.Tensor) -> torch.Tensor:
            """
            Args:
                x (torch.Tensor): a tensor with shape [1, 1, 128, 128, 128]
    
            Returns:
                torch.Tensor: a tensor with shape [1, 1, 128, 128, 128]
            """
            input = self.input(x) # [1, 1, 128, 128, 128] -> [1, 16, 128, 128, 128]
            down1_output = self.down1(input)# [1, 16, 128, 128, 128] ->[1, 32, 64, 64, 64]
            down2_output = self.down2(down1_output) # [1, 32, 64, 64, 64] -> [1, 64, 32, 32, 32]
            down3_output = self.down3(down2_output) # [1, 64, 32, 32, 32] -> [1, 128, 16, 16, 16]
            out = self.up1(down3_output, down2_output) # [1, 128, 16, 16, 16] -> [1, 64, 32, 32, 32]
            out = self.up2(out, down1_output) # [1, 64, 32, 32, 32] -> [1, 32, 64, 64, 64]
            out = self.up3(out, input) # [1, 32, 64, 64, 64] -> [1, 16, 128, 128, 128]
            out = self.output(out) # [1, 16, 128, 128, 128] -> [1, 1, 128, 128, 128]
            return out

# CBAM

from monai.networks.blocks.convolutions import Convolution
from monai.networks.layers.factories import Norm
class SingleConvolutionCBAM(nn.Module):
    """
    Auxiliary class to define a convolutional layer.
    Each convolution block: convolution, batch normalization, ReLU activation.

    Args:
        nn.Module : receive the nn.Module properties
    """
    def __init__(self, in_channels : int, out_channels : int, kernel_size: int = 3, padding: int = 1, stride : int = 1, bias: bool = False) -> None:
        """
        Args:
            in_channels (int): amount of input channels
            out_channels (int): amount of output channels
        """ 
        super(SingleConvolutionCBAM, self).__init__()
        
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.bias = bias
        
        self.singleConv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size = self.kernel_size, padding = self.padding, stride = self.stride, bias = self.bias),
            nn.BatchNorm3d(out_channels), 
            nn.ReLU(inplace=True),
        )
        
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor

        Returns:
            torch.Tensor: output tensor
        """
        return self.singleConv(x)

class DoubleConvolutionCBAM(nn.Module):
    """
    Auxiliary class to define a convolutional layer.
    Each convolution block: 3x3 convolution, batch normalization, ReLU activation.

    Args:
        nn.Module : receive the nn.Module properties
    """
    def __init__(self, in_channels : int, out_channels : int, kernel_size: int = 3, padding: int = 1, stride: int = 1, bias: bool = False) -> None:
        """
        Args:
            in_channels (int): amount of input channels
            out_channels (int): amount of output channels
        """ 
        super(DoubleConvolutionCBAM, self).__init__()
        
        self.conv1 = SingleConvolutionCBAM(in_channels, out_channels, 
                                        kernel_size=kernel_size, 
                                        padding=padding, 
                                        stride=stride, 
                                        bias=bias)
        self.conv2 = SingleConvolutionCBAM(out_channels, out_channels, 
                                        kernel_size=kernel_size, 
                                        padding=padding, 
                                        stride=stride, 
                                        bias=bias)
        
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor

        Returns:
            torch.Tensor: output tensor
        """
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class ChannelGate(nn.Module): # C x 1 x 1
    def __init__(self, in_channels: int, reduction_ratio: int = 16):
        
        super(ChannelGate, self).__init__()
        
        self.in_channels = in_channels
        self.reduction_ratio = reduction_ratio
        
        self.squeezeAvg = nn.AdaptiveAvgPool3d(1) # Global Average Pooling
        self.squeezeMax = nn.AdaptiveMaxPool3d(1) # Global Max Pooling
        self.excitation = nn.Sequential(
            nn.Linear(self.in_channels, self.in_channels // self.reduction_ratio),
            nn.ReLU(inplace=True), # ReLU activation
            nn.Linear(self.in_channels // self.reduction_ratio, self.in_channels),
        )
        self.sigActivation = nn.Sigmoid()
        
    def forward(self, x):
        
        batch_size, channels, _, _, _ = x.size()
        
        yAvg = self.squeezeAvg(x).view(batch_size, channels)
        yAvg = self.excitation(yAvg).view(batch_size, channels, 1, 1, 1)
        
        yMax = self.squeezeMax(x).view(batch_size, channels)
        yMax = self.excitation(yMax).view(batch_size, channels, 1, 1, 1)
        
        sum = yAvg + yMax
        
        return self.sigActivation(sum) * x

class SpatialGate(nn.Module): # 1 x H x W
    def __init__(self, in_channels: int, kernel_size: int = 7, padding: int = 3, bias: bool = False):
        
        super(SpatialGate, self).__init__()
        
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.padding = padding
        self.bias = bias
        
        self.squeezeAvg = nn.AdaptiveAvgPool3d(1)
        self.squeezeMax = nn.AdaptiveMaxPool3d(1)
        self.spatial = SingleConvolutionCBAM(2 * self.in_channels, self.in_channels, kernel_size = self.kernel_size, padding = self.padding)
        self.sigActivation = nn.Sigmoid()
        
    def forward(self, x):
        
        yAvg = self.squeezeAvg(x)
        yMax = self.squeezeMax(x)
        y = torch.cat([yAvg, yMax], dim=1)
        y = self.spatial(y)
        
        return self.sigActivation(y) * x

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio = 16, kernel_size: int = 7):
        
        super(CBAM, self).__init__()
        
        self.ChannelGate = ChannelGate(in_channels, reduction_ratio)
        self.SpatialGate = SpatialGate(in_channels, kernel_size = kernel_size, padding = kernel_size // 2)
        
    def forward(self, x):
        
        x_out = self.ChannelGate(x)
        x_out = self.SpatialGate(x_out)
        
        return x_out * x

class DownSamplingCBAM(nn.Module):
    """
    Auxiliary class to define a DownSamplingCBAM layer.
    Each DownSamplingCBAM block: 2x2 max pooling, double convolution and squeeze and excitation.
    input X output: [1, 16, 128, 128, 128] ->  [1, 32, 64, 64, 64] 
                    [1, 32, 64, 64, 64]    ->  [1, 64, 32, 32, 32]
                    [1, 64, 32, 32, 32]    ->  [1, 128, 16, 16, 16]

    Args:
        nn.Module: receive the nn.Module properties
    """
    def __init__(self, in_channels : int, out_channels : int, attention) -> None:
        """
        Args:
            in_channels (int): amount of input channels (16 or 32 or 64)
            out_channels (int): amount of output channels (32 or 64 or 128)
        """        
        super(DownSamplingCBAM, self).__init__()
        
        self.attentionFunction = attention
        
        self.maxpool = nn.MaxPool3d(2)
        self.conv = DoubleConvolutionCBAM(in_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False)
        self.attention = self.attentionFunction(out_channels)
        
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): _description_

        Returns:
            torch.Tensor: _description_
        """
        out = self.maxpool(x) # 2x2 max pooling -> 1/2 the size but same amount of channels
        out = self.conv(out) # double convolution -> same size but double the amount of channels
        out = self.attention(out) # squeeze and excitation
        return out

class UpSamplingCBAM(nn.Module):
    """
    Auxiliary class to define a UpSamplingCBAM layer.
    Each UpSamplingCBAM block: 2x2 UpSamplingCBAM, concatenation with skip connection, double convolution.
    input X output: [1, 128, 16, 16, 16] ->  [1, 64, 32, 32, 32]
                    [1, 64, 32, 32, 32]    ->  [1, 32, 64, 64, 64]
                    [1, 32, 64, 64, 64]    ->  [1, 16, 128, 128, 128]
                    
    Args:
        nn.Module: receive the nn.Module properties
    """
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = False) -> None:
        """
        Args:
            in_channels (int): amount of input channels (128 or 64 or 32)
            out_channels (int): amount of output channels (64 or 32 or 16)
        """
        super(UpSamplingCBAM, self).__init__()
        
        self.up = nn.ConvTranspose3d(in_channels, in_channels, kernel_size=2, stride=2)
        self.conv = DoubleConvolutionCBAM(int(in_channels + out_channels), out_channels, kernel_size=3, padding=1, stride=1, bias=False)
        
    def forward(self, x : torch.Tensor, skip_connection : torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): the input tensor
            skip_connection (torch.Tensor): the skip connection from the DownSamplingCBAM path

        Returns:
            torch.Tensor: the output tensor
        """
        x = self.up(x) # 2x2 UpSamplingCBAM -> double the size but same amount of channels
        x = torch.cat([skip_connection, x], dim=1) # concatenation with skip connection
        out = self.conv(x) # double convolution -> same size but half the amount of channels
        return out

class Attention_Unet(nn.Module):
    def __init__(self, attentionFunction, in_channels=1, out_channels=1, channels=(32, 64, 128, 128)) -> None:
        super(Attention_Unet, self).__init__()
        self.attentionFunction = attentionFunction
        layers = channels 

        self.attentionFunction = attentionFunction
        
        self.input = nn.Sequential(DoubleConvolutionCBAM(in_channels, layers[0], kernel_size=3, padding = 1, stride=1, bias=False), self.attentionFunction(layers[0])) # tranform the input to 16 channels and apply squeeze and excitation
        # encoding path
        self.down1 = DownSamplingCBAM(layers[0], layers[1], self.attentionFunction) 
        self.down2 = DownSamplingCBAM(layers[1], layers[2], self.attentionFunction) 
        self.down3 = DownSamplingCBAM(layers[2], layers[3], self.attentionFunction)
        # decoding path
        self.up1 = UpSamplingCBAM(layers[3], layers[2])
        self.up2 = UpSamplingCBAM(layers[2], layers[1])
        self.up3 = UpSamplingCBAM(layers[1], layers[0])
        self.output = nn.Sequential(nn.Conv3d(layers[0], out_channels, kernel_size=1)) # transform the output to 7 channel 
    
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): a tensor with shape [1, 1, 128, 128, 128]

        Returns:
            torch.Tensor: a tensor with shape [1, 1, 128, 128, 128]
        """
        input = self.input(x) # [1, 1, 128, 128, 128] -> [1, 16, 128, 128, 128]
        down1_output = self.down1(input)# [1, 16, 128, 128, 128] ->[1, 32, 64, 64, 64]
        down2_output = self.down2(down1_output) # [1, 32, 64, 64, 64] -> [1, 64, 32, 32, 32]
        down3_output = self.down3(down2_output) # [1, 64, 32, 32, 32] -> [1, 128, 16, 16, 16]
        out = self.up1(down3_output, down2_output) # [1, 128, 16, 16, 16] -> [1, 64, 32, 32, 32]
        out = self.up2(out, down1_output) # [1, 64, 32, 32, 32] -> [1, 32, 64, 64, 64]
        out = self.up3(out, input) # [1, 32, 64, 64, 64] -> [1, 16, 128, 128, 128]
        out = self.output(out) # [1, 16, 128, 128, 128] -> [1, 1, 128, 128, 128]
        return out



@st.cache_resource
def load_model(model_path, model_channels_tuple, model_architecture):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        if model_architecture == "3DUnet":
            model = UNet(
                spatial_dims=3,
                in_channels=1,
                out_channels=7,
                channels=model_channels_tuple,
                strides=(2, 2, 1),
                num_res_units=1,
            )

        elif model_architecture == "3DUnetAG":
            model = AttentionUnet(
                spatial_dims=3,
                in_channels=1,
                out_channels=7,
                channels=model_channels_tuple,
                strides=(2, 2, 1),
            )
            
        elif model_architecture == "3DUnetSE":
            model = SqueezeAndExcitation3DUnet(
                in_channels=1,
                out_channels=7,
                channels=model_channels_tuple,
            )
            
        elif model_architecture == "3DUnetCBAM":
            model = Attention_Unet(
                attentionFunction=CBAM,
                in_channels=1,
                out_channels=7,
                channels=model_channels_tuple,
            )

        if torch.cuda.device_count() > 1:
            st.sidebar.info(f"Menggunakan {torch.cuda.device_count()} GPUs!")
            model = nn.DataParallel(model)

        if not os.path.exists(model_path):
            st.sidebar.error(f"File model tidak ditemukan: {model_path}")
            return None, None

        # model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model = nn.DataParallel(model)
        model.to(device)
        model.load_state_dict(torch.load(model_path, weights_only=True))  
        model.eval()
        st.sidebar.success(f"Model berhasil di-load ke: {device}")
        return model, device
    except Exception as e:
        st.sidebar.error(f"Error saat me-load model: {e}")
        return None, None