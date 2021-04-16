import torch
from torch.nn import *
from tqdm import tqdm
class Convolutional(torch.nn.Module):   
    def __init__(self, features=[32, 8, 1]):
        super(Convolutional, self).__init__()

        self.convolutional_layers = Sequential(
            # Defining a 2D convolution layer
            Conv2d(1, features[0], kernel_size=3, stride=1, padding=1),
            # ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            ReLU(inplace=True),
        )

        self.linear_layers = Sequential(            
            Linear(features[0] * 8 * 8 // 4, 1),
            Sigmoid()
        )

    # Defining the forward pass    
    def forward(self, x):
        x = self.convolutional_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x

class Feedforward(torch.nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(Feedforward, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(self.hidden_size, output_size)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        hidden = self.fc1(x)
        relu = self.relu(hidden)
        output = self.fc2(relu)
        output = self.sigmoid(output)
        return output

def train_model(model, criterion, optimizer, x, y):
    optimizer.zero_grad()
    # Forward pass
    y_pred = model(x)
    # Compute Loss
    loss = criterion(y_pred.squeeze(), y)
    # Backward pass
    loss.backward()
    optimizer.step()

    predictions = (y_pred.data >= 0.5).float().flatten()
    correct_predictions = (predictions == y).sum().item()

    return loss, 100 * correct_predictions / len(y)


def test_model(model, criterion, optimizer, x, y):
    with torch.no_grad():
        # Forward pass
        y_pred = model(x)
        # Compute Loss
        loss = criterion(y_pred.squeeze(), y)
        predictions = (y_pred.data >= 0.5).float().flatten()
        correct_predictions = (predictions == y).sum().item()
        return loss, 100 * correct_predictions / len(y)

def run(model_type, options, epochs, x_train=[], x_test=[], y_train=[], y_test=[]):
    if model_type == "feedforward":
        model = Feedforward(*options)
    else:
        x_train = torch.reshape(x_train, (len(x_train), 8, 8))
        x_train = x_train.unsqueeze(1)
        x_test = torch.reshape(x_test, (len(x_test), 8, 8))
        x_test = x_test.unsqueeze(1)    
    
        model = Convolutional(*options)

    criterion = torch.nn.BCELoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    training_losses, training_accuracies = [], []
    testing_losses, testing_accuracies = [], []
    for _ in tqdm(range(epochs)):
        # model.train()
        training_loss, training_accuracy = train_model(model, criterion, optimizer, x_train, y_train)
        training_losses.append(training_loss.data)
        training_accuracies.append(training_accuracy)

        if len(x_test) > 0:        
            # model.eval()
            testing_loss, testing_accuracy = test_model(model, criterion, optimizer, x_test, y_test)
            testing_losses.append(testing_loss.data)
            testing_accuracies.append(testing_accuracy)
    torch.save(model, f"trained_{model_type}_model")
    return training_losses, testing_losses, training_accuracies, testing_accuracies
