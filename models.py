import torch

class Convolutional(torch.nn.Module):   
    def __init__(self):
        super(ConvolutionalNet, self).__init__()

        self.convolutional_layers = Sequential(
            # Defining a 2D convolution layer
            Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            #BatchNorm2d(4),
            ReLU(inplace=True),
            #MaxPool2d(kernel_size=2, stride=2),
            # Defining another 2D convolution layer
            Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            # BatchNorm2d(4),
            ReLU(inplace=True),
            # MaxPool2d(kernel_size=2, stride=2),
        )

        self.linear_layers = Sequential(
            Linear(32 * 8 * 8, 16),
            Linear(16, 1),
            Sigmoid(inplace=True)
        )

    # Defining the forward pass    
    def forward(self, x):
        x = self.conventional_layers(x)
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

    # bin_y_pred = [0 if x < 0.5 else 1 for x in y_pred]
    # accuracy = 1 - np.sum(np.square(bin_y_pred - y_test.numpy())) / len(y_test)
    predictions = (y_pred.data >= 0.5).float().flatten()
    correct_predictions = (predictions == y).sum().item()

    return loss, 100 * correct_predictions / len(y)


def test_model(model, criterion, optimizer, x, y):
    with torch.no_grad():
        # Forward pass
        y_pred = model(x)
        # Compute Loss
        loss = criterion(y_pred.squeeze(), y)
        # bin_y_pred = [0 if x < 0.5 else 1 for x in y_pred]
        # accuracy = 1 - np.sum(np.square(bin_y_pred - y_test.numpy())) / len(y_test)
        predictions = (y_pred.data >= 0.5).float().flatten()
        correct_predictions = (predictions == y).sum().item()
        return loss, 100 * correct_predictions / len(y)

def run(model_type, options, epochs, x_train=[], x_test=[], y_train=[], y_test=[]):
    model = models.Feedforward(*options)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    training_losses, training_accuracies = [], []
    testing_losses, testing_accuracies = [], []
    for _ in tqdm(range(epochs)):
        # model.train()
        training_loss, training_accuracy = models.train_model(model, criterion, optimizer, x_train, y_train)
        training_losses.append(training_loss.data)
        training_accuracies.append(training_accuracy)

        if x_test.nelement() > 0:        
            # model.eval()
            testing_loss, testing_accuracy = models.test_model(model, criterion, optimizer, x_test, y_test)
            testing_losses.append(testing_loss.data)
            testing_accuracies.append(testing_accuracy)
    return training_losses, testing_losses, training_accuracies, testing_accuracies
