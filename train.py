#!/usr/bin/python
import argparse, sys
from classifier import Training, Checkpoint

# python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
parser = argparse.ArgumentParser(description = 'Train a new classifier')
parser.add_argument('data_dir', help='directory path that contains the training data')
parser.add_argument('--save_dir', help='directory path where checkpoint.pth is saved')
parser.add_argument('--arch', help='one of torchvision.models, defaults to vgg16', default='vgg16')
parser.add_argument('--hidden_units', help='number of units in the hidden layer, defaults to 256', type=int, default=256)
parser.add_argument('--dropout', help='dropout probability, defaults to 0.5', type=float, default=0.5)
parser.add_argument('--learning_rate', help='hyperparamater to adjust during training, defaults to 0.001', type=float, default=0.001)
parser.add_argument('--epochs', help='number of epochs to run, defaults to 2', type=int, default=2)
parser.add_argument('--gpu', help='use GPU mode', action='store_true')
arguments = parser.parse_args()

training = Training(arguments)
checkpoint = training.start()
checkpoint.save(arguments.save_dir)
