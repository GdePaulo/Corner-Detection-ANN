import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from numpy import asarray


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


def get_images(path):
    images = []
    for filename in os.listdir(path):
        image = asarray(Image.open(path + filename))[:, :, 0]
        for i in range(4):
            images.append(np.rot90(image, i).flatten())
    return images


def main():
    # Define layer dimensions and dummy input
    in_features, out_features = 64, 16

    corner_images = get_images(path='images/corners/')
    no_corner_images = get_images(path='images/no_corners/')

    x_test = [np.zeros(64), np.ones(64)] + [image for image in no_corner_images] + [image for image in corner_images]
    x_test = torch.FloatTensor(x_test)
    y_test = [0, 0] + [0 for _ in range(len(no_corner_images))] + [1 for _ in range(len(corner_images))]
    y_test = torch.FloatTensor(y_test)
    x_train = x_test
    y_train = y_test

    model = Feedforward(in_features, out_features)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    model.eval()
    y_pred = model(x_test)
    before_train = criterion(y_pred.squeeze(), y_test)
    print('Test loss before training', before_train.item())

    model.train()
    epochs = 10000

    for epoch in tqdm(range(epochs)):
        optimizer.zero_grad()
        # Forward pass
        y_pred = model(x_train)
        # Compute Loss
        loss = criterion(y_pred.squeeze(), y_train)

        # print('Epoch {}: train loss: {}'.format(epoch, loss.item()))
        # Backward pass
        loss.backward()
        optimizer.step()

    model.eval()
    y_pred = model(x_test)
    print(y_pred)
    after_train = criterion(y_pred.squeeze(), y_test)
    print('Test loss after Training', after_train.item())


if __name__ == "__main__":
    # execute only if run as a script
    main()
