import torch
import torch.nn as nn

class Feedforward(torch.nn.Module):
        def __init__(self, input_size, hidden_size):
            super(Feedforward, self).__init__()
            self.input_size = input_size
            self.hidden_size  = hidden_size
            self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
            self.relu = torch.nn.ReLU()
            self.fc2 = torch.nn.Linear(self.hidden_size, 1)
            self.sigmoid = torch.nn.Sigmoid()
        def forward(self, x):
            hidden = self.fc1(x)
            relu = self.relu(hidden)
            output = self.fc2(relu)
            output = self.sigmoid(output)
            return output
            
def main():
    # Define layer dimensions and dummy input
    n_samples, in_features, out_features = 2, 3, 4
    # Make random input tensor of dimensions [n_samples, in_features]
    x = torch.randn((n_samples, in_features))


    # Create Linear layer from torch.nn module
    torch_layer = nn.Linear(in_features, out_features)

    # Load the parameters from our layer into the Pytorch layer
    torch_layer.weight = nn.Parameter(layer.weight.T) # transpose weight by .T
    torch_layer.bias = nn.Parameter(layer.bias)

    # Perform forward pass
    torch_y = torch_layer(x)
    
    print(torch_y)
    print("corners")


if __name__ == "__main__":
    # execute only if run as a script
    main()