# encoding: utf-8

"""
The main CheXpert models implementation.
Including:
    DenseNet-121
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from . import densenet


class SelfAttentionInit(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttentionInit, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)

        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)

        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma*out + x
        return out
    
    

class MaskLayer(nn.Module):
    def __init__(self):
        super(MaskLayer, self).__init__()
        self.mask = nn.Parameter(torch.rand(1024,7,7))
        
        
    def forward(self, x):
        #x_min = self.mask.amin(dim=(-3,-2,-1), keepdim=True)
        #x_max = self.mask.amax(dim=(-3,-2,-1), keepdim=True)
        #norm_mask = (self.mask - x_min) / (x_max - x_min)
        
        masked = x * self.mask
        return masked
'''
class MaskLayer(nn.Module):
    def __init__(self, attention_weights):
        super(MaskLayer, self).__init__()
        self.mask = nn.Parameter(attention_weights)
        #self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        masked = x * self.mask
        return masked
'''
class DenseNet121(nn.Module):
    """Model modified.
    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.
    """
    def __init__(self, out_size, mode, drop_rate=0):
        super(DenseNet121, self).__init__()
        assert mode in ('U-Ones', 'U-Zeros', 'U-MultiClass')
        self.densenet121 = densenet.densenet121(pretrained=True, drop_rate=drop_rate)
        num_ftrs = self.densenet121.classifier.in_features ##1024
        
        if mode in ('U-Ones', 'U-Zeros'):
            
            #self.attention_model = SelfAttentionInit(1024)
            #self.mask_layer = MaskLayer(self.attention_model(torch.rand(1,1024,7,7)))
            
            self.mask_layer = MaskLayer()
            self.densenet121.classifier_mask = nn.Sequential(
                nn.Linear(num_ftrs, out_size),
                #nn.Sigmoid()
            )
            self.densenet121.classifier_nonmask = nn.Sequential(
                nn.Linear(num_ftrs, out_size),
                #nn.Sigmoid()
            )
        elif mode in ('U-MultiClass', ):
            self.densenet121.classifier = None
            self.densenet121.Linear_0 = nn.Linear(num_ftrs, out_size)
            self.densenet121.Linear_1 = nn.Linear(num_ftrs, out_size)
            self.densenet121.Linear_u = nn.Linear(num_ftrs, out_size)
            
        self.mode = mode
        
        # Official init from torch repo.
        for m in self.densenet121.modules():
            if isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

        self.drop_rate = drop_rate
        self.drop_layer = nn.Dropout(p=drop_rate)

    def forward(self, x):
        features = self.densenet121.features(x)
        out = F.relu(features, inplace=True)
        
        ############Chuankai Xu Change#################
        '''
        initial_mask = self.attention_model(out)
        mask_part = self.mask_layer(initial_mask)
        '''
        mask_part = self.mask_layer(out)
        non_mask_part = out - mask_part
        
        out_mask1 = F.adaptive_avg_pool2d(mask_part, (1, 1)).view(features.size(0), -1)
        out_nonmask1 = F.adaptive_avg_pool2d(non_mask_part, (1, 1)).view(features.size(0), -1)
        
        ###########End ###############
        if self.drop_rate > 0:
            out_mask = self.drop_layer(out_mask1)
            out_nonmask = self.drop_layer(out_nonmask1)
            
        self.activations = out_mask
        
        #print(self.activations.shape)
        
        
        if self.mode in ('U-Ones', 'U-Zeros'):
            mask_out = self.densenet121.classifier_mask(out_mask)
            non_mask_out = self.densenet121.classifier_nonmask(out_nonmask)
            non_mask_out = F.softmax(non_mask_out, dim=1)
            
        #############End
        elif self.mode in ('U-MultiClass', ):
            n_batch = x.size(0)
            out_0 = self.densenet121.Linear_0(out).view(n_batch, 1, -1)
            out_1 = self.densenet121.Linear_1(out).view(n_batch, 1, -1)
            out_u = self.densenet121.Linear_u(out).view(n_batch, 1, -1)
            out = torch.cat((out_0, out_1, out_u), dim=1)
            
        return non_mask_part, mask_part, non_mask_out, mask_out, self.activations

class DenseNet161(nn.Module):
    """Model modified.
    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.
    """
    def __init__(self, out_size, mode, drop_rate=0):
        super(DenseNet161, self).__init__()
        assert mode in ('U-Ones', 'U-Zeros', 'U-MultiClass')
        self.densenet161 = densenet.densenet161(pretrained=True, drop_rate=drop_rate)
        num_ftrs = self.densenet161.classifier.in_features
        if mode in ('U-Ones', 'U-Zeros'):
            self.densenet161.classifier = nn.Sequential(
                nn.Linear(num_ftrs, out_size),
                #nn.Sigmoid()
            )
        elif mode in ('U-MultiClass', ):
            self.densenet161.classifier = None
            self.densenet161.Linear_0 = nn.Linear(num_ftrs, out_size)
            self.densenet161.Linear_1 = nn.Linear(num_ftrs, out_size)
            self.densenet161.Linear_u = nn.Linear(num_ftrs, out_size)
            
        self.mode = mode
        
        # Official init from torch repo.
        for m in self.densenet161.modules():
            if isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

        self.drop_rate = drop_rate
        self.drop_layer = nn.Dropout(p=drop_rate)

    def forward(self, x):
        features = self.densenet161.features(x)
        out = F.relu(features, inplace=True)
        
        out = F.adaptive_avg_pool2d(out, (1, 1)).view(features.size(0), -1)
        
        if self.drop_rate > 0:
            out = self.drop_layer(out)
        self.activations = out
        
        if self.mode in ('U-Ones', 'U-Zeros'):
            out = self.densenet161.classifier(out)
        elif self.mode in ('U-MultiClass', ):
            n_batch = x.size(0)
            out_0 = self.densenet161.Linear_0(out).view(n_batch, 1, -1)
            out_1 = self.densenet161.Linear_1(out).view(n_batch, 1, -1)
            out_u = self.densenet161.Linear_u(out).view(n_batch, 1, -1)
            out = torch.cat((out_0, out_1, out_u), dim=1)
            
        return self.activations, out