import torch.nn as nn
import torch


class Model(nn.Module):
    def __init__(self, layer_sizes):
        """
        Initializes the model with a flexible layer structure.
        Args:
            layer_sizes (list of int): A list where each integer is the number of
                                       neurons in a hidden layer.
        """
        super().__init__()
        self.layers = nn.ModuleList()
        
        all_layer_dims = [1] + layer_sizes + [1]
        
        for i in range(len(all_layer_dims) - 1):
            in_features = all_layer_dims[i]
            out_features = all_layer_dims[i+1]
            self.layers.append(nn.Linear(in_features, out_features))

    def forward(self, x, return_neuron_outputs=False):
        neuron_outputs = []
        
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))
            if return_neuron_outputs:
                neuron_outputs.append(x)
                
        x = self.layers[-1](x)
        
        if return_neuron_outputs:
            return x, neuron_outputs
        return x
