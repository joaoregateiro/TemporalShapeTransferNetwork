from __future__ import print_function
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, num_points = 6890):
        super(Encoder, self).__init__()
        
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)

        # Temporal model
        self.input = nn.Linear(3*6890, 1024)
        self.lstm = nn.LSTM(1024, 256, 3)
        self.out = nn.Linear(256, 3*6890)

    def forward(self, x):
        batch_size = x.size(0)

        t_x = x.reshape(batch_size, 1, -1)
        t_x = F.relu(self.input(t_x))
        output, _ = self.lstm(t_x, None)
        output = F.tanh(self.out(output))

        output = output.reshape(batch_size, 3, -1)


        x = self.conv1(x)
        x = F.leaky_relu(x)

        x = self.conv2(x)
        x = F.leaky_relu(x)

        x = self.conv3(x)
        x = F.leaky_relu(x)

        x = torch.cat((x, output), 1)

        return x

    def forward2(self, x):

        x = self.conv1(x)
        x    = F.leaky_relu(x)

        x = self.conv2(x)
        x    = F.leaky_relu(x)

        x = self.conv3(x)
        x    = F.leaky_relu(x)

        return x

    def forward1(self, x):
        x = x.transpose(1, 2).transpose(0, 1)

        x, _ = self.conv1(x,None)
        x    = F.leaky_relu(x)

        x, _ = self.conv2(x,None)
        x    = F.leaky_relu(x)

        x, _ = self.conv3(x,None)
        x    = F.leaky_relu(x)
        x = x.transpose(0, 1).transpose(1, 2)
        return x


class AdaIN(nn.Module):
    def __init__(self,norm,input_nc,planes):
        super(AdaIN,self).__init__()
        self.conv_weight = nn.Conv1d(input_nc, planes, 1)
        self.conv_bias = nn.Conv1d(input_nc, planes, 1)
        self.norm = norm(planes)
    
    def forward(self,x,addition):

        x = self.norm(x)

        weight = self.conv_weight(addition)
        bias = self.conv_bias(addition)
        out =  weight * x + bias

        return out

class AdaINResBlock(nn.Module):
    def __init__(self,input_nc,planes,norm=nn.InstanceNorm1d,conv_kernel_size=1,padding=0):
        super(AdaINResBlock,self).__init__()
        self.adain1 = AdaIN(norm=norm,input_nc=input_nc,planes=planes)
        self.lrelu = nn.ReLU()
        self.conv1 = nn.Conv1d(planes, planes, kernel_size=conv_kernel_size, stride=1, padding=padding)
        self.adain2 = AdaIN(norm=norm,input_nc=input_nc,planes=planes)
        self.conv2 = nn.Conv1d(planes,planes,kernel_size=conv_kernel_size, stride=1, padding=padding)
        self.adain_res = AdaIN(norm=norm,input_nc=input_nc,planes=planes)
        self.conv_res=nn.Conv1d(planes,planes,kernel_size=conv_kernel_size, stride=1, padding=padding)

    def forward(self,x,addition):

        out = self.adain1(x,addition)
        out = self.lrelu(out)
        out = self.conv1(out)
        out = self.adain2(out,addition)
        out = self.lrelu(out)
        out = self.conv2(out)

        residual = x
        residual = self.adain_res(residual,addition)
        residual = self.lrelu(residual)
        residual = self.conv_res(residual)

        out = out + residual

        return  out


class Decoder(nn.Module):
    def __init__(self, bottleneck_size = 1024):
        self.bottleneck_size = bottleneck_size
        super(Decoder, self).__init__()

        self.conv1 = torch.nn.Conv1d(self.bottleneck_size, self.bottleneck_size, 1) 
        self.conv2 = torch.nn.Conv1d(self.bottleneck_size, self.bottleneck_size//2, 1) 
        self.conv3 = torch.nn.Conv1d(self.bottleneck_size//2, self.bottleneck_size//4, 1) 
        self.conv4 = torch.nn.Conv1d(self.bottleneck_size//4, 3, 1) 

        self.adain_block1 = AdaINResBlock(input_nc=3 ,planes=self.bottleneck_size)
        self.adain_block2 = AdaINResBlock(input_nc=3 ,planes=self.bottleneck_size//2)
        self.adain_block3 = AdaINResBlock(input_nc=3 ,planes=self.bottleneck_size//4)
        
        self.norm1 = torch.nn.InstanceNorm1d(self.bottleneck_size)
        self.norm2 = torch.nn.InstanceNorm1d(self.bottleneck_size//2)
        self.norm3 = torch.nn.InstanceNorm1d(self.bottleneck_size//4)
        self.th = nn.Tanh()
        

    def forward(self, x, addition):

        x = self.norm1(self.conv1(x))
        x = self.adain_block1(x,addition)
        x = self.norm2(self.conv2(x))
        x = self.adain_block2(x,addition)
        x = self.norm3(self.conv3(x))
        x = self.adain_block3(x,addition)  

        x = 2*self.th(self.conv4(x))

        return x


class Discriminator(nn.Module):
    def __init__(self, input_size = 20670):
        super(Discriminator, self).__init__()

        self.fc1 = nn.Linear(input_size, 512)  
        self.fc2 = nn.Linear(512, 128) # hidden layer

        self.adv_layer = nn.Sequential(nn.Linear(128, 1), nn.Sigmoid())

        self.lrelu = nn.ReLU()

        self.th = nn.Tanh()

    def forward(self, x):
        batch_size = x.shape[0]

        x = (self.lrelu(self.fc1(x)))
        x = (self.lrelu(self.fc2(x)))

        x = x.view(batch_size, -1)

        return self.adv_layer(x)


class StyleShapeTransferGenerator(nn.Module):
    def __init__(self, num_points = 6890, bottleneck_size = 1027):
        super(StyleShapeTransferGenerator, self).__init__()
        self.num_points = num_points
        self.bottleneck_size = bottleneck_size
        self.encoder = Encoder(num_points = num_points)
        self.decoder = Decoder(bottleneck_size = self.bottleneck_size+3)

    def forward(self, x1, x2):

        #x2 = self.encoder(x2)
        y   = torch.cat((x1, x2), 1)

        out = self.decoder(y,x2)

        return out.transpose(2,1)
