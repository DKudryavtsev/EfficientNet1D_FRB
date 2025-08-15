#########1#########2#########3#########4#########5#########6#########7#########
"""1D convolutional neural networks for multichannel time series classification"
   (1) A simple fully convolutional network: FCN1d
   (2) A network based on the EfficientNet family: EfficientNet1dXS   
"""
import torch
import torch.nn as nn


class FCN1d(nn.Module):
    """
    1D fully convolutional network
    Input shape: [B, C, L]
    Output shape: [B], logits of binary classification

    Args:
        inchan (int): # input channels
        activation (class, optional): activation function.
                                      Defaults to torch.nn.GELU.
    """
    def __init__(self, inchan, activation=nn.GELU):
        super().__init__()
        self.activation = activation()
        self.model = nn.Sequential(
            nn.Conv1d(inchan, 128, 7, padding='same'),
            nn.BatchNorm1d(128),
            self.activation,
            nn.Conv1d(128, 256, 5, padding='same'),
            nn.BatchNorm1d(256),
            self.activation,
            nn.Conv1d(256, 128, 3, padding='same'),
            nn.BatchNorm1d(128),
            self.activation,
            nn.AdaptiveAvgPool1d(1), # output: [B, 128 ,1]
            nn.Conv1d(128, 1, 1),    # output: [B, 1, 1]
            nn.Flatten(0)
        )
    
    def forward(self, x):
        x = self.model(x)
        return x


class Downsample1d(nn.Module):
    """
    Downsampler of a residual connection.
    Use this for differing number of channels or size
    of the output of a residual block.
    
    Args:
        inchan (int): # input channels
        outchan (int): # output channels
        stride (int, optional): stride. Defaults to 1.
    """
    def __init__(self, inchan, outchan, stride=1):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv1d(inchan, outchan, 1, stride, bias=False),
            nn.BatchNorm1d(outchan)
        )

    def forward(self, x):
        x = self.model(x)
        return x
    

class SE1d(nn.Module):
    """
    Squeeze-and-excitation block (1D)

    Args:
        inchan (int): # input channels
        reduced (int): # hidden channels, recommendation is 1/16 of the input
        activation (class): activation function
    """
    def __init__(self, inchan, reduced, activation):
        super().__init__()
        self.activation = activation()
        self.scale = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(inchan, reduced, 1),
            self.activation,
            nn.Conv1d(reduced, inchan, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        scale = self.scale(x)  # [B, C ,1]
        x = scale * x          # [B, C, L]
        return x
    

class StochasticDepth1d(nn.Module):
    """
    This module randomly turns off a residual block during training
    by zeroing its output (only residual connection passes the information)

    Args:
        p (float, optional): Probability to "survive". Defaults to 0.8.
    """
    def __init__(self, p=0.8):
        super().__init__()
        self.p =  p

    def forward(self, x):
        if not self.training:
            return x
        
        binary_tensor = torch.rand(x.shape[0], 1, 1, device=x.device) < self.p
        x = x * binary_tensor
        return x
    

class MBConv1d(nn.Module):
    """
    Mobile inverted residual bottleneck convolution (1D)

    Args:
        inchan (int): # input channels
        outchan (int): # output channels
        kernel (int): kernel size
        stride (int): stride
        expansion (int): expansion factor
        activation (class): activation function 
        fused (bool, optional): fused or not. Defaults to False.
    """
    def __init__(
            self, inchan, outchan, kernel, stride, expansion, 
            activation, fused=False):
        super().__init__()
        self.activation = activation()        
        hidden = inchan * expansion
        #se_reduced = inchan // 4   # Original EfficientNet
        #se_reduced = outchan // 4  # More logical
        se_reduced = hidden // 16   # Advice of the SE block authors
        
        # Residual shortcut
        if (inchan == outchan) and (stride == 1):
            self.residual = nn.Identity()
        else:
            self.residual = Downsample1d(inchan, outchan, stride)
        
        # Padding (aliquot divisions)
        padding = (kernel-1) // 2
        
        # convolutional block
        layers = []
        if fused:
            layers.extend([
                nn.Conv1d(inchan, hidden, kernel, stride, padding, bias=False),
                nn.BatchNorm1d(hidden),
                self.activation
            ])
        else:
            layers.extend([
                nn.Conv1d(inchan, hidden, 1, bias=False),
                nn.BatchNorm1d(hidden),
                self.activation,
                nn.Conv1d(
                    hidden, hidden, kernel, stride, padding,
                    groups=hidden, bias=False),
                nn.BatchNorm1d(hidden),
                self.activation,
                SE1d(hidden, se_reduced, activation),
            ])
        layers.extend([
            nn.Conv1d(hidden, outchan, 1, bias=False),
            nn.BatchNorm1d(outchan),
            StochasticDepth1d()
        ])
        self.convblock = nn.Sequential(*layers)

    def forward(self, x):
        residual = self.residual(x)
        x = self.convblock(x)
        x += residual
        return x
    

class EfficientNet1dXS(nn.Module):
    """
    EfficientNet 1D extra small 
    Input shape: [B, C, L]
    Output shape: [B], logits of binary classification

    Args:
        inchan (int): # input channels
        out_classes (int): # output classes
        activation (class, optional): activation function.
                                      Defaults to torch.nn.GELU.
    """
    def __init__(self, inchan, out_classes, activation=nn.GELU):
        super().__init__()
        self.activation = activation()
        self.backbone = nn.Sequential(
            nn.Conv1d(inchan, 12, 3, 2, 1, bias=False),
            nn.BatchNorm1d(12),
            self.activation,
            
            # inchan, outchan, kernel, stride, expansion, ...
            MBConv1d(12, 12, 3, 1, 1, activation, fused=True),

            MBConv1d(12, 24, 3, 2, 2, activation, fused=True),
            MBConv1d(24, 24, 3, 1, 2, activation, fused=True),

            MBConv1d(24, 32, 3, 2, 2, activation, fused=True),
            MBConv1d(32, 32, 3, 1, 2, activation, fused=True),

            MBConv1d(32, 64, 3, 2, 2, activation),
            MBConv1d(64, 64, 3, 1, 2, activation),
            MBConv1d(64, 64, 3, 1, 2, activation),

            MBConv1d(64, 80, 3, 1, 3, activation),
            MBConv1d(80, 80, 3, 1, 3, activation),
            MBConv1d(80, 80, 3, 1, 3, activation),
            MBConv1d(80, 80, 3, 1, 3, activation),

            MBConv1d(80, 128, 3, 2, 3, activation),
            MBConv1d(128, 128, 3, 1, 3, activation),
            MBConv1d(128, 128, 3, 1, 3, activation),
            MBConv1d(128, 128, 3, 1, 3, activation),
            MBConv1d(128, 128, 3, 1, 3, activation),
            MBConv1d(128, 128, 3, 1, 3, activation),
            MBConv1d(128, 128, 3, 1, 3, activation),
            
            nn.Conv1d(128, 640, 1, bias=False),
            nn.AdaptiveAvgPool1d(1), # [B, 640, 1]
            nn.Flatten()
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.2, inplace=True),
            nn.Linear(640, out_classes),
        )
    
    def forward(self, x):
        x = self.backbone(x)
        x= self.classifier(x)
        return x