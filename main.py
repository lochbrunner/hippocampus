#!/usr/bin/env python3

import logging

from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
import argparse


logger = logging.getLogger(__name__)


class Activation(nn.Module):
    def forward(self, x):
        return F.relu6(x*10)/6


class Hippocampus(nn.Module):
    def hook(self, i, m, inputs, outputs):
        x = outputs[0].detach().cpu().numpy().flatten()
        if self.neurons[i] is None:
            self.neurons[i] = x
        else:
            self.neurons[i] += x

    def reset(self):
        for neuron in self.neurons:
            neuron.fill(0.)

    def __init__(self, *layers):
        super(Hippocampus, self).__init__()

        def hook_wrapper(module, i):
            def do(*args):
                return self.hook(i, *args)
            module.register_forward_hook(do)
            return None

        for layer in layers:
            layer.weight.data = torch.from_numpy(
                np.random.choice((0., 1.), layer.weight.shape, p=(0.9, 0.1))).type(torch.FloatTensor)

        self.neurons = [hook_wrapper(module, i)
                        for i, module in enumerate(layers[1:])]

        self.modules = [l for layer in layers for l in [
            layer, Activation()]]

    def forward(self, x):
        for module in self.modules:
            x = module(x)
        return x


def predict(model, data, neuron_groups):
    model.reset()
    model(data)
    neurons = np.concatenate(model.neurons)

    activations = np.array([neurons[neuron_group].sum()
                            for neuron_group in neuron_groups])

    return np.argsort(-activations)


class Tops:
    def __init__(self):
        self.tops = [0]*10

    def __add__(self, arg):
        (predictions, gt) = arg
        for i, prediction in enumerate(predictions):
            if prediction == gt:
                self.tops[i] += 1
                return self

    def __str__(self):
        total = sum(self.tops)
        return f'error: {(1.-self.tops[0]/total)*100:2f}%'


def main(args):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    logger.info('Downloading')
    training_dataset = datasets.MNIST(
        './data', train=True, download=True, transform=transform)
    testing_dataset = datasets.MNIST(
        './data', train=False, transform=transform)

    train_loader = torch.utils.data.DataLoader(
        training_dataset, shuffle=False, batch_size=1)
    test_loader = torch.utils.data.DataLoader(
        testing_dataset, shuffle=False, batch_size=1)

    model = Hippocampus(
        nn.Conv2d(1, 32, 3, 1),
        nn.Conv2d(32, 32, 3, 1),
        nn.Conv2d(32, 32, 3, 1),
        nn.Conv2d(32, 32, 3, 1),
        nn.Conv2d(32, 32, 3, 1),
        nn.Conv2d(32, 32, 3, 1),
    )

    # Train
    neuron_groups = []
    for i in range(10):
        for data, target in tqdm(train_loader, leave=False, desc=f'Digit: {i}'):
            if i != target.item():
                continue
            model(data)

        neurons = np.concatenate(model.neurons)
        p = np.quantile(neurons, args.quantile)
        neuron_groups.append(np.argwhere(neurons > p))
        model.reset()

    # test
    tops = Tops()
    for data, target in tqdm(test_loader, leave=False, desc='Testing'):
        p = predict(model, data, neuron_groups)
        tops += (p, target.item())

    print(tops)
    print(tops.tops)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--quantile', type=float, default=0.95)
    loggers = [logging.getLogger(name)
               for name in logging.root.manager.loggerDict]

    for logger in loggers:
        logger.setLevel(logging.INFO)
    main(parser.parse_args())
