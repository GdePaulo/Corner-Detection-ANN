import os
from os import path
import pickle
import torch
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm
from numpy import asarray
from sklearn.model_selection import StratifiedKFold


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


def draw_pieslice(draw, img_size, angle, rotation, width):
    offset = 40
    shape = [(0-offset, 0-offset), (img_size+offset, img_size+offset)]

    draw.pieslice(shape, start=rotation, end=angle+rotation, outline="white", width=width)


def draw_shapes(draw, width):
    # We should eventually work to an automated shape generator
    # shape_coordinates = [[(4, 4), (20, 20)], [(30, 4), (50, 20)], [(4, 30), (20, 50)], [(30, 30), (50, 50)]]
    # corner_coordinates = []
    #
    # for coordinate in shape_coordinates:
    #     coordinates = draw_polygon(draw, coordinate[0], coordinate[1])
    #     if coordinates:
    #         corner_coordinates.append(coordinates)

    corner_coordinates = [draw_rectangle(draw, width, (4, 4), (20, 20)),
                          draw_triangle(draw, width, (30, 4), (30, 30), (56, 30)),
                          draw_trapezium(draw, width, (80, 4), (100, 4), (116, 20), (64, 20)),
                          draw_octagon(draw, width, (50, 60), (67, 43), (91, 43), (108, 60))]
    draw_line(draw, width, (4, 30), (20, 30))
    draw_circle(draw, width, (4, 80), (40, 116))

    return [item for sublist in corner_coordinates for item in sublist]


def rotate_shape(shape, rotation):
    print()


# No corner shapes

def draw_line(draw, width, left, right):
    draw.line((left, right), fill="white", width=width)
    return


def draw_circle(draw, width, left, right):
    draw.ellipse((left, right), outline="white", width=width)
    return


# Corner shapes

def draw_rectangle(draw, width, top_left, bottom_right):
    draw.rectangle((top_left, bottom_right), outline="white", width=width)
    top_right = (bottom_right[0], top_left[1])
    bottom_left = (top_left[0], bottom_right[1])
    return top_left, top_right, bottom_left, bottom_right


def draw_triangle(draw, width, c1, c2, c3):
    draw.line((c1, c2, c3, c1), fill="white", width=width)
    return c1, c2, c3


def draw_trapezium(draw, width, c1, c2, c3, c4):
    draw.line((c1, c2, c3, c4, c1), fill="white", width=width)
    return c1, c2, c3, c4


def draw_octagon(draw, width, c1, c2, c3, c4):
    segment_length = c3[0] - c2[0]
    across_length = c4[0] - c1[0]
    c5, c6, c7, c8 = (c4[0], c4[1] + segment_length), (c3[0], c3[1] + across_length), (c2[0], c2[1] + across_length), \
                     (c1[0], c1[1] + segment_length)
    draw.line((c1, c2, c3, c4, c5, c6, c7, c8, c1), fill="white", width=width)
    return c1, c2, c3, c4, c5, c6, c7, c8


def wizard(img, img_size, corners, corner_images, no_corner_images):
    sliding_window = 8

    # corner_images_count = 0
    # no_corner_images_count = 0

    for i in range(img_size - sliding_window):
        for j in range(img_size - sliding_window):
            sub_img = img.crop((i, j, i + sliding_window, j + sliding_window))
            is_corner = False
            for corner in corners:
                if i + (sliding_window / 4) <= corner[0] < i + sliding_window - (sliding_window / 4) and + \
                        j + (sliding_window / 4) <= corner[1] < j + sliding_window - (sliding_window / 4):
                    is_corner = True
                    break

            img_array = 1 - np.clip(np.array(sub_img), 0, 1)
            img_array = img_array.flatten()

            if is_corner:
                # corner_images_count += 1
                if not any(np.array_equal(img_array, x) for x in corner_images):
                    corner_images.append(img_array)
            else:
                # no_corner_images_count += 1
                if not any(np.array_equal(img_array, x) for x in no_corner_images):
                    no_corner_images.append(img_array)

    # print(len(corner_images))
    # print(corner_images_count)
    # print(len(no_corner_images))
    # print(no_corner_images_count)


def train_model(model, criterion, optimizer, x_train, x_test, y_train, y_test):
    y_pred = model(x_test)
    before_train = criterion(y_pred.squeeze(), y_test)

    model.train()
    epochs = 100000

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

    bin_y_pred = [0 if x < 0.5 else 1 for x in y_pred]
    percentage = 1 - np.sum(np.square(bin_y_pred - y_test.numpy())) / len(y_test)

    return before_train.item(), after_train.item(), percentage

    # print('Test loss after Training', after_train.item())
    # Print the labelling of the trained model.
    # model_labelling_result = zip(y_test, y_pred)
    # for i, label in enumerate(model_labelling_result):
    #     print(
    #         f'Expected {label[0]} : Predicted {label[1].data}. [{"Corner" if label[0] == 1 else "No Corner"}]'
    #         f'[{"True" if label[0] == int(torch.round(label[1])) else "False"}]')
    #     if label[0] == 0 and label[1].data > 0.85:
    #         print(x_train[i].reshape((8, 8)))
    #     elif label[0] == 1 and label[1].data < 0.1:
    #         print(x_train[i].reshape((8, 8)))


def main():
    # Define layer dimensions.
    # Input layer is of size 64 (8x8 kernel).
    # Hidden layer is of size 16 (neurons).
    # Output layer is of size 1 (probability of an edge in the given 8x8 sub-image).
    in_features, hidden_features, out_features = 64, 16, 1

    corner_images, no_corner_images = [], []

    if path.exists("corner_images.pickle") and path.exists("no_corner_images.pickle"):
        corner_images = pickle.load(open("corner_images.pickle", "rb"))
        no_corner_images = pickle.load(open("no_corner_images.pickle", "rb"))
    else:
        for rotation in range(360)[::6]:
            for angle in [45, 90, 135]:
                for width in [1, 2, 3]:
                    img_size = 21
                    img = Image.new("L", (img_size, img_size))
                    draw = ImageDraw.Draw(img)
                    draw_pieslice(draw, img_size, angle, rotation, width)
                    wizard(img, img_size, [(int(img_size/2), int(img_size/2))], corner_images, no_corner_images)

        print(f'Amount of corner images: {len(corner_images)}')
        print(f'Amount of no corner images: {len(no_corner_images)}')

        for width in [1, 2, 3]:
            img_size = 128
            img = Image.new("L", (img_size, img_size))
            draw = ImageDraw.Draw(img)
            corners = draw_shapes(draw, width)
            img.save(f"test_{width}.png")
            wizard(img, img_size, corners, corner_images, no_corner_images)

        print(f'Amount of corner images: {len(corner_images)}')
        print(f'Amount of no corner images: {len(no_corner_images)}')

        corner_images += get_images(path='images/corners/')
        no_corner_images += get_images(path='images/no_corners/')

    print(f'Amount of corner images: {len(corner_images)}')
    print(f'Amount of no corner images: {len(no_corner_images)}')

    with open('corner_images.pickle', 'wb') as handle:
        pickle.dump(corner_images, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('no_corner_images.pickle', 'wb') as handle:
        pickle.dump(no_corner_images, handle, protocol=pickle.HIGHEST_PROTOCOL)

    x = [image for image in no_corner_images] + [image for image in corner_images]
    x = torch.FloatTensor(x)
    y = [0 for _ in range(len(no_corner_images))] + [1 for _ in range(len(corner_images))]
    y = torch.FloatTensor(y)

    skf = StratifiedKFold(n_splits=10)
    skf.get_n_splits(x, y)
    losses_before = np.array([])
    losses_after = np.array([])
    percentages = np.array([])
    for train_index, test_index in skf.split(x, y):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model = Feedforward(in_features, hidden_features, out_features)
        criterion = torch.nn.BCELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        model.eval()
        before, after, percentage = train_model(model, criterion, optimizer, x_train, x_test, y_train, y_test)
        percentages = np.append(percentages, percentage)
        losses_before = np.append(losses_before, before)
        losses_after = np.append(losses_after, after)

    for i in range(len(losses_before)):
        print(f'{losses_before[i]} vs {losses_after[i]} with accuracy of: {percentages[i]}')
    print(f'Mean before: {np.mean(losses_before)} vs mean after: {np.mean(losses_after)} with a mean accuracy of {np.mean(percentages)}')


if __name__ == "__main__":
    # execute only if run as a script
    main()
