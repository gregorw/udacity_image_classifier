#!/usr/bin/python
import argparse, sys
from classifier import Predictor, Checkpoint
from PIL import Image

# python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
parser = argparse.ArgumentParser(description = 'Train a new classifier')
parser.add_argument('input_image', help='path to image file')
parser.add_argument('checkpoint', help='path to checkpoint file')
parser.add_argument('--category_names', help='mapping of categories to real names')
parser.add_argument('--gpu', help='use GPU mode', action='store_true')
args = parser.parse_args()

checkpoint = Checkpoint.load(args.checkpoint, args.gpu)
image = Image.open(args.input_image)
predictor = Predictor(checkpoint, args.category_names, args.gpu)
name, confidence = predictor.perform(image)
print('Predicted ‘{}’ with confidence {:.3f}'.format(name, confidence))
