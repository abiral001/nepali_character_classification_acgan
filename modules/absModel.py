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
    def __init__(self, classes = 59):
        super(absDis, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5)
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5)
        )
        self.layer6 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5)
        )
        self.realOrFake=nn.Linear(4*4*512, 1)
        self.whichDigit=nn.Linear(4*4*512, classes)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()
        
    def forward(self, x):
        x = x.float()
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = x.view(-1, 4*4*512)
        realOrFake = self.sigmoid(self.realOrFake(x))
        whichDigit = self.softmax(self.whichDigit(x))
        return realOrFake, whichDigit

        # activation = bias * input + weight 