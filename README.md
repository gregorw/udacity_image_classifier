# AI Programming with Python Project

Project code for Udacity's AI Programming with Python Nanodegree program. In this project, students first develop code for an image classifier built with PyTorch, then convert it into a command line application.

## Usage
Get help with the following commands:

> python train.py -h
> python predict.py -h

### Tested examples

**Training**

> python train.py flowers --gpu --epochs 1 --arch vgg13 --hidden_units 124
Checkpoint written to ./1547998999_checkpoint.pth

> python train.py flowers --gpu --epochs 1 --arch vgg11 --hidden_units 124 --dropout 0.5
Checkpoint written to ./1548006356_checkpoint.pth

**Prediction**

> python predict.py flowers/test/23/image_03440.jpg 1547998999_checkpoint.pth --gpu
Predicted ‘23’ with confidence 1.000

> python predict.py flowers/test/23/image_03442.jpg 1547998999_checkpoint.pth --gpu --category_names cat_to_name.json
Predicted ‘fritillary’ with confidence 1.000

> python predict.py flowers/test/24/image_06815.jpg 1547998999_checkpoint.pth --gpu
Predicted ‘24’ with confidence 0.658

> python predict.py flowers/test/24/image_06815.jpg 1547998999_checkpoint.pth --gpu --category_names cat_to_name.json
Predicted ‘red ginger’ with confidence 0.658

> python predict.py flowers/test/10/image_07104.jpg test/1547991223_checkpoint.pth --gpu
Predicted ‘10’ with confidence 0.999

> python predict.py flowers/test/10/image_07104.jpg test/1547991223_checkpoint.pth --gpu --category_names cat_to_name.json
Predicted ‘globe thistle’ with confidence 0.999

> python predict.py flowers/test/90/image_04405.jpg test/1547991223_checkpoint.pth --gpu --category_names cat_to_name.json
Predicted ‘canna lily’ with confidence 0.935

> python predict.py flowers/test/90/image_04405.jpg test/1547991223_checkpoint.pth --gpu
Predicted ‘90’ with confidence 0.935

> python predict.py flowers/test/23/image_03440.jpg 1548006356_checkpoint.pth 
Predicted ‘23’ with confidence 0.988

> python predict.py flowers/test/23/image_03440.jpg 1548006356_checkpoint.pth --gpu
Predicted ‘23’ with confidence 0.988
