import torch
import torch.nn as nn
from torch.nn.modules.activation import Softmax
from torch.nn.modules.batchnorm import BatchNorm1d, BatchNorm2d

class absGenerator(nn.Module):
    def __init__(self, in_channels):
        super(absGenerator, self).__init__()
        self.l1 = nn.Linear(in_channels, 128)
        self.l2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(5,5), stride=2, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.l3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(5,5), stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.l4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=8, kernel_size=(5,5), stride=2, padding=2),
            nn.BatchNorm2d(8),
            nn.ReLU()
        )
        self.l5 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=8, out_channels=3, kernel_size=(3,3), stride=2, padding=2),
            nn.BatchNorm2d(3),
            nn.ReLU()
        )
        self.l6 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=3, out_channels=1, kernel_size=(2,2), stride=1, padding=2),
            nn.Tanh()
        )
        
    def forward(self, x):
        x = x.view(-1, 110)
        x = self.l1(x)
        x = x.view(-1, 128, 1, 1)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        return x

class absDiscriminator(nn.Module):
    def __init__(self, classes = 59):
        super(absDiscriminator, self).__init__()
        self.c1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5)
        )
        self.c2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5)
        )
        self.c3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5)
        )
        self.c4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5)
        )
        self.c5 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5)
        )
        self.c6 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5)
        )
        self.fc_source=nn.Linear(4*4*512, 1)
        
        self.fc_class=nn.Linear(4*4*512, classes)
        
        self.sig = nn.Sigmoid()
        
        self.soft = nn.LogSoftmax(dim=1)
        
    def forward(self, x):
        x = x.float()
        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)
        x = self.c4(x)
        x = self.c5(x)
        x = self.c6(x)
        x = x.view(-1, 4*4*512)
        rf = self.sig(self.fc_source(x))
        c = self.soft(self.fc_class(x))
        return rf, c