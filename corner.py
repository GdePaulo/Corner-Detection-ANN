import os
from os import path
import pickle
import torch
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm
from numpy import asarray
from sklearn.model_selection import StratifiedKFold
import data
import models
import random


def get_highlighted_corners(model, image_path):
    img = Image.open(image_path)
    img = img.convert("L")
    width, height = img.size

    highlighted_corners_img = np.zeros((width, height))

    sliding_window = 8
    for i in range(width - sliding_window):
        for j in range(height - sliding_window):
            sub_img = img.crop((i, j, i + sliding_window, j + sliding_window))
            img_array = np.clip(np.array(sub_img), 0, 1).flatten()
            if model(torch.FloatTensor(img_array)) >= 0.5:
                for _i in range(4):
                    for _j in range(4):
                        highlighted_corners_img[i+2+_i][j+2+_j] += 1

    highlighted_corners_img = np.flip(np.rot90(highlighted_corners_img, 3), 1)
    highlighted_corners_img *= 255.0 / highlighted_corners_img.max()
    highlighted_corners_img = Image.fromarray(np.uint8(highlighted_corners_img))
    highlighted_corners_img.save(f"{image_path}_output.png")

def main(validation, train, corner_detection):
    # Define layer dimensions.
    # Input layer is of size 64 (8x8 kernel).
    # Hidden layer is of size 16 (neurons).
    # Output layer is of size 1 (probability of an edge in the given 8x8 sub-image).
    in_features, hidden_features, out_features = 64, 16, 1
    x, y = data.generate_data() 
    epochs = 7500

    if validation:
        # run_training_and_testing("feedforward", [in_features, hidden_features, out_features], epochs, x, y)
        # run_training_and_testing("convolutional", [], epochs, x, y)

        for n in [100, 500, 1000, 1500, 2155]:
            indices = torch.randperm(len(x))
            sub_x = torch.index_select(x, 0, indices[:n])
            sub_y = torch.index_select(y, 0, indices[:n])
            # run_validation("feedforward", [in_features, hidden_features, out_features], epochs, sub_x, sub_y)
            run_validation("convolutional", [], epochs, sub_x, sub_y)
    elif train:
        # training_losses, _, training_accuracies, _ = models.run("feedforward", [in_features, hidden_features, out_features], epochs=epochs, x_train=x, y_train=y)
        training_losses, _, training_accuracies, _ = models.run("convolutional", [], epochs=epochs, x_train=x, y_train=y)
        print(f'training losses \n start: {training_losses[0]} end: {training_losses[-1]} min: {min(training_losses)}')
        print(f'training accuracies \n start: {training_accuracies[0]} end: {training_accuracies[-1]} min: {max(training_accuracies)}')
    elif corner_detection:
        model = torch.load("trained_feedforward_model")
        model.eval()
        get_highlighted_corners(model=model, image_path="image.png")


def run_validation(model_type, options, epochs, x, y):
    skf = StratifiedKFold(n_splits=10, shuffle=True)
    skf.get_n_splits(x, y)

    best_training_losses = np.array([])
    best_testing_losses = np.array([])
    best_training_accuracies = np.array([])
    best_testing_accuracies = np.array([])
    
    last_training_losses = np.array([])
    last_testing_losses = np.array([])
    last_training_accuracies = np.array([])
    last_testing_accuracies = np.array([])
    
    for train_index, test_index in skf.split(x, y):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        training_losses, testing_losses, training_accuracies, testing_accuracies = models.run(model_type, options, epochs=epochs, x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)

        # training_losses, testing_losses, training_accuracies, testing_accuracies = models.run("convolutional", [], epochs=epochs, x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)

        best_training_losses = np.append(best_training_losses, min(training_losses))
        best_testing_losses = np.append(best_testing_losses, min(testing_losses))
        best_training_accuracies = np.append(best_training_accuracies, max(training_accuracies))
        best_testing_accuracies = np.append(best_testing_accuracies, max(testing_accuracies))

        last_training_losses = np.append(last_training_losses, training_losses[-1])
        last_testing_losses = np.append(last_testing_losses, testing_losses[-1])
        last_training_accuracies = np.append(last_training_accuracies, training_accuracies[-1])
        last_testing_accuracies = np.append(last_testing_accuracies, testing_accuracies[-1])
    
    for i in range(len(best_testing_losses)):
        print(f'minimum losses {i}\n training {best_training_losses[i]} vs testing {best_testing_losses[i]}')
        print(f'maximum accuracies {i}\n training {best_training_accuracies[i]} vs testing {best_testing_accuracies[i]}')
        
        print(f'last losses {i}\n training {last_training_losses[i]} vs testing {last_testing_losses[i]}')
        print(f'last accuracies {i}\n training {last_training_accuracies[i]} vs testing {last_testing_accuracies[i]}')

    k_data = [[] for x in range(5)]
    if  path.exists(f"k_data_validation_{model_type}.pickle"):
        k_data = pickle.load(open(f"k_data_validation_{model_type}.pickle", "rb"))
    
    with open(f'k_data_validation_{model_type}.pickle', 'wb') as handle:
        k_data[0].append(np.mean(last_training_losses))
        k_data[1].append(np.mean(last_testing_losses))
        k_data[2].append(np.mean(last_training_accuracies))
        k_data[3].append(np.mean(last_testing_accuracies))
        k_data[4].append(len(y))
        pickle.dump(k_data, handle, protocol=pickle.HIGHEST_PROTOCOL)



def run_training_and_testing(model_type, options, epochs, x, y):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    skf.get_n_splits(x, y)

    train_index, test_index = next(skf.split(x, y))
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]

    training_losses, testing_losses, training_accuracies, testing_accuracies = models.run(model_type, options, epochs=epochs, x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)
    print(f'minimum losses \n training {min(training_losses)} vs testing {min(testing_losses)}')
    print(f'maximum accuracies \n training {max(training_accuracies)} vs testing {max(testing_accuracies)}')
    
    print(f'last losses \n training {training_losses[-1]} vs testing {testing_losses[-1]}')
    print(f'last accuracies \n training {training_accuracies[-1]} vs testing {testing_accuracies[-1]}')
    
    k_data = [[] for x in range(5)]
    
    with open(f'k_data_training_{model_type}.pickle', 'wb') as handle:
        k_data[0] = training_losses
        k_data[1] = testing_losses
        k_data[2] = training_accuracies
        k_data[3] = testing_accuracies
        k_data[4] = [x + 1 for x in range(len(training_losses))]
        pickle.dump(k_data, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main(validation=True, train=True, corner_detection=True)
