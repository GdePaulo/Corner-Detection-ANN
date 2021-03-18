import torch


class Feedforward(torch.nn.Module):

    def __init__(self, input_size, hidden_size):
        super(Feedforward, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
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
    n_samples, in_features, out_features = 2, 64, 1
    # Make random input tensor of dimensions [n_samples, in_features]
    x = torch.randn((n_samples, in_features))

    print("corners")


if __name__ == "__main__":
    # execute only if run as a script
    main()
