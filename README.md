# Object classification without gradient descent

Inspired by the Hippocampus.

Finding corelation in the network.

## Method

1. Feed each digit from the training dataset separately to the network and sum the neuron activations.
2. Find the neurons with the highest activations and store their indices (neuron group)
3. Feed a sample from the test dataset and check which neuron group has the highest sum of activations.
