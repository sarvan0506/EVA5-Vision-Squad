# import necessary libraries
from __future__ import print_function
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, bn="None", dropout=0, num_splits=4):
        """ This function instantiates all the model layers """
        super(Net, self).__init__()

        self.convblock1 = self.conv2d(1, 8, 3, bn, dropout, num_splits) # Input: 28x28x1 | Output: 26x26x8 | RF: 3x3

        self.convblock2 = self.conv2d(8, 8, 3, bn, dropout, num_splits) # Input: 26x26x8 | Output: 24x24x8 | RF: 5x5

        self.convblock3 = self.conv2d(8, 16, 3, bn, dropout, num_splits) # Input: 24x24x8 | Output: 22x22x16 | RF: 7x7

        self.pool = nn.MaxPool2d(2, 2)  # Input: 22x22x16 | Output: 11x11x16 | RF: 8x8
        
        self.convblock4 = self.conv2d(16, 16, 3, bn, dropout, num_splits) # Input: 11x11x16 | Output: 9x9x16 | RF: 12x12

        self.convblock5 = self.conv2d(16, 16, 3, bn, dropout, num_splits) # Input: 9x9x16 | Output: 7x7x16 | RF: 16x16

        self.convblock6 = self.conv2d(16, 16, 3, bn, dropout, num_splits) # Input: 7x7x16 | Output: 5x5x16 | RF: 20x20

        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1)
        )  # Input: 5x5x10 | Output: 1x1x10 | RF: 28x28
    
    def conv2d(self, in_channels, out_channels, kernel_size, bn, dropout, num_splits):
        
        if bn == "BN":
            conv = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size),
                nn.ReLU(),
                BatchNorm(out_channels),
                nn.Dropout(dropout)
            )
        elif bn == "GBN":
            conv = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size),
                nn.ReLU(),
                GhostBatchNorm(out_channels, num_splits),
                nn.Dropout(dropout)
            )
        else:
            conv = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        
        return conv
    
    
    def forward(self, x):
        """ This function defines the network structure """
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.pool(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        return F.log_softmax(x, dim=-1)
