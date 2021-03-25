import os
import sys

import torch
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm
from numpy import asarray


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


def get_images(path):
    images = []
    # Loop through all images in the given directory.
    for filename in os.listdir(path):
        image = asarray(Image.open(path + filename))[:, :, 0]
        # Flipping zeros and ones because the lines from the input images are black instead of white.
        # Clipping the image to values of 0 or 1.
        image = 1 - np.clip(image, 0, 1)
        # Generating the 4 possible rotations of the input image.
        for i in range(4):
            images.append(np.rot90(image, i).flatten())
    return images


def draw_shapes(draw):
    shape_coordinates = [[(4, 4), (20, 20)], [(30, 4), (50, 20)], [(4, 30), (20, 50)], [(30, 30), (50, 50)]]
    corner_coordinates = []

    for coordinate in shape_coordinates:
        corner_coordinates.append(draw_rectangle(draw, coordinate[0], coordinate[1]))

    return [item for sublist in corner_coordinates for item in sublist]


def draw_rectangle(draw, top_left, bottom_right):
    draw.rectangle((top_left, bottom_right), outline="white")
    top_right = (bottom_right[0], top_left[1])
    bottom_left = (top_left[0], bottom_right[1])
    return top_left, top_right, bottom_left, bottom_right


def draw_line(draw, left, right):
    print()


def draw_ellipse(draw, left, right):
    print()


def draw_polygon(draw, left, right):
    print()


def wizard():
    img_size = 64
    img = Image.new("L", (img_size, img_size))
    draw = ImageDraw.Draw(img)
    corners = draw_shapes(draw)
    img.save("test.png")
    sliding_window = 8

    corner_images = []
    no_corner_images = []

    for i in range(img_size - sliding_window):
        for j in range(img_size - sliding_window):
            sub_img = img.crop((i, j, i + sliding_window, j + sliding_window))
            is_corner = False
            for corner in corners:
                if i + (sliding_window / 4) <= corner[0] < i + sliding_window - (sliding_window / 4) and +\
                        j + (sliding_window / 4) <= corner[1] < j + sliding_window - (sliding_window / 4):
                    is_corner = True
                    break

            if is_corner:
                corner_images.append(sub_img)
            else:
                no_corner_images.append(sub_img)

    corner_images[11].save("corner.png")
    no_corner_images[11].save("no_corner.png")
    print(len(corner_images))
    print(len(no_corner_images))


def main():
    # Define layer dimensions.
    # Input layer is of size 64 (8x8 kernel).
    # Hidden layer is of size 16 (neurons).
    # Output layer is of size 1 (probability of an edge in the given 8x8 sub-image).
    in_features, hidden_features, out_features = 64, 16, 1

    corner_images = get_images(path='images/corners/')
    no_corner_images = get_images(path='images/no_corners/')

    x_test = [np.zeros(64), np.ones(64)] + [image for image in no_corner_images] + [image for image in corner_images]
    x_test = torch.FloatTensor(x_test)
    y_test = [0, 0] + [0 for _ in range(len(no_corner_images))] + [1 for _ in range(len(corner_images))]
    y_test = torch.FloatTensor(y_test)
    x_train = x_test
    y_train = y_test

    model = Feedforward(in_features, hidden_features, out_features)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    model.eval()
    y_pred = model(x_test)
    before_train = criterion(y_pred.squeeze(), y_test)
    print('Test loss before training', before_train.item())

    model.train()
    epochs = 10000

    for _ in tqdm(range(epochs)):
        optimizer.zero_grad()
        # Forward pass
        y_pred = model(x_train)
        # Compute Loss
        loss = criterion(y_pred.squeeze(), y_train)
        # Backward pass
        loss.backward()
        optimizer.step()

    model.eval()
    y_pred = model(x_test)
    after_train = criterion(y_pred.squeeze(), y_test)
    print('Test loss after Training', after_train.item())

    # Print the labelling of the trained model.
    model_labelling_result = zip(y_test, y_pred)
    for label in model_labelling_result:
        print(
            f'Expected {label[0]} : Predicted {int(label[1])}. [{"Corner" if label[0] == 1 else "No Corner"}]'
            f'[{"True" if label[0] == int(torch.round(label[1])) else "False"}]')


if __name__ == "__main__":
    # execute only if run as a script
    # main()
    wizard()
