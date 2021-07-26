#!/usr/bin/env python3

# memorize the neurons with highest activations
# sum all activated neurons per connection

from torchvision import datasets
import PIL

training_dataset = datasets.MNIST('./data', train=True)

for i in range(10): 
    image =training_dataset[i][0] 
    PIL.ImageOps.invert(image).save(f'docs/image.{i}.png', format='png')