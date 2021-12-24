import torch
import torch.nn as nn
from torch.nn.modules.activation import Softmax
from torch.nn.modules.batchnorm import BatchNorm1d, BatchNorm2d

class absGen(nn.Module):
    def __init__(self, random_noise, kernel_size_conv):
        super(absGen, self).__init__()
        self.random_noise = random_noise
        self.kernel_size_conv = kernel_size_conv
        self.layer1 = nn.Linear(in_features=self.random_noise, out_features = 768)
        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size=self.kernel_size_conv, stride = 2, padding = 2),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
        )
        self.conv_layer3 = nn.Sequential(
            nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size=self.kernel_size_conv, stride = 2, padding = 2),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        ) 
        self.conv_layer4 = nn.Sequential(
            nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size=self.kernel_size_conv, stride = 2, padding = 2),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
        )
        self.conv_layer5 = nn.Sequential(
            nn.Conv2d(in_channels = 128, out_channels = 96, kernel_size=self.kernel_size_conv, stride = 2, padding = 2),
            nn.BatchNorm2d(96),
            nn.ReLU(True),
        )
        self.conv_layer6 = nn.Sequential(
            nn.Conv2d(in_channels = 96, out_channels = 3, kernel_size=self.kernel_size_conv, stride = 2, padding = 2),
            nn.BatchNorm2d(3),
            nn.Tanh(),
        )
    
    def forward(self, inp):
        inp2 = self.layer1(inp)
        inp3 = self.conv_layer2(inp2)
        inp4 = self.conv_layer3(inp3)
        inp5 = self.conv_layer4(inp4)
        inp6 = self.conv_layer5(inp5)
        inp7 = self.conv_layer6(inp6)

        return inp7

class absDis(nn.Module):
    def __init__(self, image_size, kernel_size_conv, flatten):
        super(absDis, self).__init__()
        self.image_size = image_size
        self.Layer1 = nn.Linear(self.image_size, out_features=2)
        self.conv_layer2 = nn.Sequential(
            nn.conv2d(in_features = 1, out_features = 2, kernel_size = self.kernel_size_conv, stride = 2, padding = 2),
            nn.BatchNorm2d(2),
            nn.ReLU(True),  
        )
        self.conv_layer3 = nn.Sequential(
            nn.conv2d(in_features = 2, out_features = 16, kernel_size = self.kernel_size_conv, stride = 2, padding = 2),
            nn.BatchNorm2d(16),
            nn.ReLU(True),   
        )
        self.conv_layer4 = nn.Sequential(
            nn.conv2d(in_features = 16, out_features = 32, kernel_size = self.kernel_size_conv, stride = 2, padding = 2),
            nn.BatchNorm2d(32),
            nn.ReLU(True),  
        )
        self.conv_layer5 = nn.Sequential(
            nn.conv2d(in_features = 32, out_features = 8, kernel_size = self.kernel_size_conv, stride = 2, padding = 2),
            nn.BatchNorm2d(8),
            nn.ReLU(True),  
        )
        self.conv_layer6 = nn.Sequential(
            nn.conv2d(in_features = 8, out_features = 3, kernel_size = self.kernel_size_conv, stride = 2, padding = 2),
            nn.BatchNorm2d(3),
            nn.Tanh(),   ##Tanh  for image 
        )

        self.flatten = nn.Flatten(self.conv_layer6)
        self.fcnn_layer8 = nn.Linear(in_features = 3, out_channel = 2+36+14)
        self.sigg = nn.ReLU(self.fcnn_layer8)


    def forward(self, input_x):
        input_1 = nn.layer1(input_x)
        input_2 = nn._conv_layer2(input_1)
        input_3 = nn._conv_layer3(input_2)
        input_4 = nn._conv_layer4(input_3)
        input_5 = nn._conv_layer5(input_4)
        input_6 = nn._conv_layer6(input_5)
        return input_6

        # activation = bias * input + weight 


  