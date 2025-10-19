# NN structures
# Tao Hong, Zhaoyi Xu, Se Young Chun, Luis Hernandez-Garcia, and Jeffrey A. Fessler, 
# ``Convergent Complex Quasi-Newton Proximal Methods for Gradient-Driven Denoisers in Compressed Sensing MRI Reconstruction'',
# To appear in IEEE Transactions on Computational Imaging, arXiv:2505.04820, 2025.

import torch
import torch.nn as nn
import torch.nn.functional as F

class SmoothActivation(nn.Module):
    def __init__(self):
        super(SmoothActivation, self).__init__()
        self.elu = nn.ELU()  # ELU is the smooth activation specified
        self.Softplus = nn.Softplus(beta=100.0)
    def forward(self, x):
        return self.elu(x)

class ResidualBlock(nn.Module):
    def __init__(self, channels,isBais=True):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1,bias=False)
        if isBais:
            self.conv2 = nn.Conv2d(1, channels, kernel_size=3, stride=1, padding=1,bias=True)
        else:
            self.conv2 = nn.Conv2d(1, channels, kernel_size=3, stride=1, padding=1,bias=False)
        self.smooth_activation = SmoothActivation()
    
    def forward(self,x,x_input):
        x = self.smooth_activation(self.conv1(x)+self.conv2(x_input))
        return x 
        
class Network(nn.Module):
    def __init__(self, num_blocks=5,isBais=True):
        super(Network, self).__init__()
        if isBais:
            self.initial_conv = nn.Conv2d(1, 128, kernel_size=3, stride=1, padding=1,bias=True)
        else:
            self.initial_conv = nn.Conv2d(1, 128, kernel_size=3, stride=1, padding=1,bias=False)  # Initial Conv layer with 128 channels
        self.blocks = nn.ModuleList([ResidualBlock(128,isBais) for _ in range(num_blocks)])  # Residual blocks
        self.smooth_activation = SmoothActivation()

        # Final convolution and global average pooling
        self.final_conv_128 = nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1,bias=False)  # Last layer 128 -> 1 channels
        self.final_conv_1 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1,bias=False)  # One more Conv layer with 1 channel
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # Global Average Pooling
    
    def forward(self, x):
        x_input = x
        z = self.smooth_activation(self.initial_conv(x_input))
        
        # Apply the sequence of residual blocks
        for block in self.blocks:
            z = block(z,x_input)

        # Final convolution layers and global average pooling
        x = self.global_avg_pool(self.final_conv_128(z)+self.final_conv_1(x_input))
        #x = self.final_conv_128(z)+self.final_conv_1(x_input)
        return x.view(x.size(0), -1)  # Flatten to (batch_size, 1)
    

class NetworkDenoise(nn.Module):
    def __init__(self, num_blocks=5,isBais=True):
        super(NetworkDenoise, self).__init__()
        if isBais:
            self.initial_conv = nn.Conv2d(1, 128, kernel_size=3, stride=1, padding=1,bias=True)
        else:
            self.initial_conv = nn.Conv2d(1, 128, kernel_size=3, stride=1, padding=1,bias=False)  # Initial Conv layer with 128 channels
        self.blocks = nn.ModuleList([ResidualBlock(128,isBais) for _ in range(num_blocks)])  # Residual blocks
        self.smooth_activation = SmoothActivation()

        # Final convolution and global average pooling
        self.final_conv_128 = nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1,bias=False)  # Last layer 128 -> 1 channels
        self.final_conv_1 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1,bias=False)  # One more Conv layer with 1 channel
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # Global Average Pooling
    
    def forward(self, x):
        x_input = x
        z = self.smooth_activation(self.initial_conv(x_input))

        # Apply the sequence of residual blocks
        for block in self.blocks:
            z = block(z,x_input)

        # Final convolution layers and global average pooling
        x = self.final_conv_128(z)+self.final_conv_1(x_input)
        return x
    