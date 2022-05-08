import os
from argparse import ArgumentParser

from modules.absNormalize import ABSNormalize
from modules.absParams import ABSparams
from modules.absTraining import ABStraining

if __name__ == '__main__':
    parser = ArgumentParser(description="Nepali Character Recognition using ACGAN")
    parser.add_argument('-t', '--training', type=str, default="Training")
    parser.add_argument('-m', '--machine', type=str, default='cpu')
    args = parser.parse_args()
    if args.training == "Training":
        mod = ABStraining.startTraining()
    else:
        print("Not optimized")
