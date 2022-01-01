import os
from argparse import ArgumentParser

from modules.absNormalize import ABSNormalize
from modules.absModel import absGen, absDis
from modules.absParams import ABSparams

if __name__ == '__main__':
    parser = ArgumentParser(description='Nepali Character Recognition using AC GAN')
    parser.add_argument('-m', '--mode', type=str, default='training')
    parser.add_argument('-d', '--device', type=str, default='CPU')
    parser.add_argument('-s', '--source', type=str, help="Data Images used for training")
    args = parser.parse_args()

    


    