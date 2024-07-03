import torch
import torch.nn as nn
import torch.nn.functional as F

from models import register
from argparse import Namespace

class DConv(nn.Module):
    def __init__(self, in_channels, out_channel, kernel_size, stride=1, padding='same', bias=False):
        super(DConv, self).__init__()

        self.dConv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=1, groups=in_channels, padding=padding, bias=bias),
            nn.Conv2d(in_channels, out_channel, kernel_size=1)
            )
        
    def forward(self, x):
        x = self.dConv(x)
        
        return x
    
class DenseLayer(nn.Module):
    def __init__(self, in_channelss, out_channels, bias=False):
        super(DenseLayer, self).__init__()
        self.conv = nn.Conv2d(in_channelss, out_channels, kernel_size=(3, 3), padding="same", bias=bias)
        self.ReLU = nn.ReLU(inplace=True)
        
        
    def forward(self, x):
        return torch.cat([x, self.ReLU(self.conv(x))], 1)
    
class RDB(nn.Module):
    def __init__(self, in_channelss, growth_rate, num_layers, bias=False):
        super(RDB, self).__init__()
        self.layers = nn.Sequential(*[DenseLayer(in_channelss + growth_rate * i, growth_rate, bias=bias) for i in range(num_layers)])
        self.lff = nn.Conv2d(in_channelss + growth_rate * num_layers, growth_rate, kernel_size=1, bias=bias)
        
    def forward(self, x):
        return x + self.lff(self.layers(x))

class RDN(nn.Module):
    def __init__(self, num_channels, num_features, growth_rate, num_blocks, num_layers, bias=False):
        super(RDN, self).__init__()
        self.num_blocks = num_blocks
        
        self.shallowF1 = nn.Conv2d(num_channels, num_features, kernel_size=7, padding="same", bias=bias)
        self.shallowF2 = nn.Conv2d(num_features, num_features, kernel_size=7, padding="same", bias=bias)
        
        self.rdbs = nn.ModuleList([RDB(num_features, growth_rate, num_layers)])
        
        for _ in range(num_blocks - 1):
            self.rdbs.append(RDB(growth_rate, growth_rate, num_layers))
            
        self.gff = nn.Sequential(
            nn.Conv2d(growth_rate*num_blocks, num_features, kernel_size=1, bias=bias),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding="same", bias=bias)
            )
        
    def forward(self, x):        
        sfe1 = self.shallowF1(x)
        sfe2 = self.shallowF2(sfe1)
        
        x = sfe2
        local_features = []
        for i in range(self.num_blocks):
            x = self.rdbs[i](x)
            local_features.append(x)
 
        x = self.gff(torch.cat(local_features, 1)) + sfe1
        
        return x

class AutoEncoder(nn.Module):
    def __init__(self, in_channels, out_channel, expansion=4):
        super(AutoEncoder, self).__init__()
                
        self.conv_in = nn.Conv2d(in_channels, expansion*in_channels, kernel_size=7, stride=1, padding='same', bias=False)
        self.encoder = nn.Sequential(
            DConv(expansion*in_channels, expansion*in_channels, kernel_size=7),         
            nn.PixelUnshuffle(2),
            nn.ReLU(inplace=True),
            DConv(expansion*in_channels*2**2, expansion*in_channels, kernel_size=7),         
            nn.PixelUnshuffle(2),
            nn.ReLU(inplace=True),
            )      
        
        self.decoder = nn.Sequential(
            DConv(expansion*in_channels*2**2, expansion*in_channels*2**2, kernel_size=7),         
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True),
            DConv(expansion*in_channels, expansion*in_channels*2**2, kernel_size=7),         
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True),
            )
        
        self.GA = nn.Sequential(self.encoder,
                                self.decoder,
                                )
        
        self.conv_out = nn.Conv2d(expansion*in_channels, out_channel, kernel_size=3, stride=1, padding='same', bias=False)
        
    def forward(self, x):
        _, _, h, w = x.size()
        # Check if height is not divisible by 16 (2^2)
        if h % 4 != 0:
            x = nn.functional.pad(x, (0, 0, 0, 4 - h % 4))
        
        # Check if width is not divisible by 16 (2^2)
        if w % 4 != 0:
            x = nn.functional.pad(x, (0, 4 - w % 4, 0, 0))
        
        conv_in = self.conv_in(x)
        out = self.GA(conv_in)
        out = conv_in + out
        
        out = self.conv_out(out)
        
        return out

class DPCA(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DPCA, self).__init__()
        self.global_avg_pool_h = nn.AdaptiveAvgPool2d((1, None))
        self.global_avg_pool_v = nn.AdaptiveAvgPool2d((None, 1))
        self.conv_h = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.conv_v = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.conv_f = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.out_channels = out_channels

    def forward(self, x):
        h = self.global_avg_pool_h(x)
        v = self.global_avg_pool_v(x)
        h = self.conv_h(h)
        v = self.conv_v(v)
        f = self.conv_f(x)
        h = self.sigmoid(h)
        v = self.sigmoid(v)
        f = self.sigmoid(f)
        
        return h * v * f
 
class TFAM(nn.Module):
    def __init__(self, channels_in, num_features=256, bias=False):
        super(TFAM, self).__init__()
        self.bias=bias
        
        self.convIn = nn.Sequential(DConv(channels_in, num_features, kernel_size=3, bias=self.bias),
                                    nn.ReLU(inplace=True),
                                    )
        self.DPCA = DPCA(num_features, 2*num_features)
        self.posAvg = nn.AvgPool2d(2)
        self.posMax = nn.MaxPool2d(2)
        self.POS_unit = self.POS(num_features)        
        self.CA_unit = self.CA(num_features)  
        
        
        
        self.convOut = nn.Sequential(
            DConv(2*num_features, channels_in, kernel_size=3, bias=self.bias)
            )
    
    def POS(self, channels_in):
        block = nn.Sequential(
            DConv(2*channels_in, channels_in*(2**2), kernel_size=3, bias=self.bias),   
            nn.PixelShuffle(2),
            DConv(channels_in, 2*channels_in, kernel_size=3, bias=self.bias),
            )
        
        return block
    
    def CA(self, channels_in):
        block = nn.Sequential(
            nn.Conv2d(channels_in, 2*channels_in, kernel_size=1, stride=1, groups=channels_in, padding='same', bias=self.bias),
            nn.PixelUnshuffle(2),
            DConv(2*channels_in*(2**2), 2*channels_in*(2**2), kernel_size=3, bias=self.bias),
            nn.PixelShuffle(2),
            DConv(2*channels_in, channels_in, kernel_size=3, bias=self.bias),
        )
        
        return block

    def forward(self, x):
        convIn = self.convIn(x)
        dpca = self.DPCA(convIn)
        out = self.convOut((torch.cat((self.CA_unit(convIn), convIn), dim=1) +
                           self.POS_unit(torch.cat((self.posAvg(convIn), self.posMax(convIn)), dim=1)))*dpca)
        
        #out = torch.cat((torch.sigmoid(out), x), dim=1)
        out = x*torch.sigmoid(out) 
        
        return out
           
class AdaptiveResidualBlock(nn.Module):
    def __init__(self, in_channels, growth_rate = 12, bias=False):
        super(AdaptiveResidualBlock, self).__init__()
        self.growth_rate = in_channels*growth_rate
        
        self.BNpath = nn.Sequential(
            nn.Conv2d(in_channels, growth_rate, kernel_size=3, stride=1, padding=1, bias=bias, dilation=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=bias, dilation=1),
            TFAM(growth_rate),
            nn.ReLU(inplace=True),
            nn.Conv2d(growth_rate, in_channels, kernel_size=3, stride=1, padding=1, bias=bias, dilation=1),
            )
        
        self.ADPpath = nn.Sequential(
            nn.AvgPool2d(2),
            nn.Conv2d(in_channels, in_channels*4, kernel_size=1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.PixelShuffle(2)
            )
        
        self.convOut = nn.Conv2d(in_channels, in_channels, kernel_size=3, groups=in_channels, stride=1, padding=1, bias=bias)
        
    def forward(self, x):
        resPath = self.convOut(self.BNpath(x) + x)
        outADP = self.ADPpath(x)
        
        out = resPath + outADP
        
        return out
    
class ARC(nn.Module):
    def __init__(self, channels_in, expansion = 128):
        super(ARC, self).__init__()
        self.input = AdaptiveResidualBlock(channels_in, expansion)
        
    def forward(self, x):
        return self.input(x)

class ResidualConcatenationBlock(nn.Module):
    def __init__(self, channels_in, out_channels, num_layers=3, bias=False):
        super(ResidualConcatenationBlock, self).__init__()
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.pw = nn.ModuleList([nn.Conv2d((2**(i+1))*channels_in, (2**(i+1))*channels_in, kernel_size=1,
                                stride=1, padding=0, bias=bias)
                                  for i in range(num_layers-1)])
        self.pw.append(nn.Conv2d(2**(num_layers)*channels_in, out_channels, kernel_size=1,
                                  stride=1, padding=0, bias=bias))
        
        self.block = nn.ModuleList([ARC((2**i)*channels_in)
                                  for i in range(num_layers)])

    def forward(self, x):
        for i in range(1, self.num_layers+1):
            x = torch.cat((self.block[i-1](x), x), dim=1)
            x = self.pw[i-1](x)
        x = nn.Conv2d(x.shape[1], self.out_channels, kernel_size=1, stride=1, padding=0, bias=False).cuda()(x) #
        
        return x

class ResidualModule(nn.Module):
    def __init__(self, channels_in, out_channels, num_layers=2, bias=False):
        super(ResidualModule, self).__init__()
        self.num_layers = num_layers
        self.pw = nn.ModuleList([nn.Conv2d((2**(i+1))*channels_in, (2**(i+1))*channels_in, kernel_size=1, stride=1, padding='same', bias=bias)
                                  for i in range(num_layers-1)])
        self.pw.append(nn.Conv2d(2**(num_layers)*channels_in, out_channels, kernel_size=1, stride=1, padding='same', bias=bias))        
        
        self.block = nn.ModuleList([ResidualConcatenationBlock((2**i)*channels_in, (2**i)*channels_in)
                                  for i in range(num_layers)])

    def forward(self, x):
        for i in range(self.num_layers):
            x = torch.cat((self.block[i](x), x), dim=1)
            pw = self.pw[i](x)

        return pw
    
class FeatureModule(nn.Module):
    def __init__(self, channels_in, skip_connection_channels):
        super(FeatureModule, self).__init__()
        self.TFAM = TFAM(channels_in)
        self.conv =  nn.Conv2d(channels_in, skip_connection_channels, kernel_size=3, stride=1, padding='same', bias=False)
        
    def forward(self, x, skip_connection):
        out = self.conv(self.TFAM(x))
        output = torch.add(out, skip_connection)
        
        return output

class Cgnet(nn.Module):
    def __init__(self, args):
        super(Cgnet, self).__init__()
        self.inputLayer = nn.Sequential(AutoEncoder(3, 128),
                                        RDN(128, 128, 128, 32, 3)
                                        )
        self.RM = ResidualModule(128, 128)
        self.FM = FeatureModule(128, 128)
        self.RDN = RDN(128, 128, 128, 18, 3)
        
        self.conv1 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding='same', bias=False)
        
        self.upscale = []
        for _ in range(2 // 2):
            self.upscale.extend([
                nn.Conv2d(128, 128*2**2, kernel_size=3, stride=1, padding='same', bias=False),
                nn.PixelShuffle(2)
                ])
            
        self.upscale = nn.Sequential(*self.upscale)
        self.Output = nn.Conv2d(128, 3, kernel_size=3, stride=1, padding='same', bias=False)
        
    def forward(self, x):
        orig_h, orig_w = 2*x.size(-2), 2*x.size(-1)
        Input = self.inputLayer(x)
        RMOutput = self.RM(Input)
        FMOutput = self.FM(RMOutput, Input)
        PS1 = self.conv1(FMOutput)
        PS1 = self.upscale(PS1)
        PS1 = self.RDN(PS1)
    
        output = self.Output(PS1)
        output = output[:, :, :orig_h, :orig_w]
        output = output + F.interpolate(x, (orig_h, orig_w), mode='nearest')
        
        return output
             
@register('cgnet')        
def cgnet(in_channels, out_channels):
    args = Namespace()
    args.in_channels = in_channels
    args.out_channels = out_channels
    
    return Cgnet(args)