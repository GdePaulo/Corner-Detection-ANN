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
    
    epochs = 1000

    print("I just dated ")

    if validation:
        skf = StratifiedKFold(n_splits=10, shuffle=True)
        skf.get_n_splits(x, y)

        best_training_losses = np.array([])
        best_testing_losses = np.array([])
        best_training_accuracies = np.array([])
        best_testing_accuracies = np.array([])

        for train_index, test_index in skf.split(x, y):
            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]
            training_losses, testing_losses, training_accuracies, testing_accuracies = models.run("feedforward", [in_features, hidden_features, out_features], epochs=epochs, x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)
            best_training_losses = np.append(best_training_losses, min(training_losses))
            best_testing_losses = np.append(best_testing_losses, min(testing_losses))
            best_training_accuracies = np.append(best_training_accuracies, max(training_accuracies))
            best_testing_accuracies = np.append(best_testing_accuracies, max(testing_accuracies))
        
        for i in range(len(best_testing_losses)):
            print(f'minimum losses {i}\n training {best_training_losses[i]} vs testing {best_testing_losses[i]}')
            print(f'maximum accuracies {i}\n training {best_training_accuracies[i]} vs testing {best_testing_accuracies[i]}')
        # print(
        #     f'Mean before: {np.mean(losses_before)} vs mean after: {np.mean(losses_after)} with a mean accuracy of {np.mean(percentages)}')
    elif train:
      
        training_losses, _, training_accuracies, _ = models.run("feedforward", [in_features, hidden_features, out_features], epochs=epochs, x_train=x, y_train=y)
        # # print(f'{before} vs {after} with accuracy of: {percentage}')
        print(f'training losses \n start: {training_losses[0]} end: {training_losses[-1]} min: {min(training_losses)}')
        # print(f'testing losses \n start: {testing_losses[0]} end: {testing_losses[-1]} min: {min(testing_losses)}')
        
        print(f'training accuracies \n start: {training_accuracies[0]} end: {training_accuracies[-1]} min: {max(training_accuracies)}')
        # print(f'testing accuracies \n start: {testing_accuracies[0]} end: {testing_accuracies[-1]} min: {max(testing_accuracies)}')
        # print(f'testing accuracies \n{testing_accuracies==training_accuracies}')

        # torch.save(model, "trained_model")
    elif corner_detection:
        model = torch.load("trained_model")
        model.eval()
        get_highlighted_corners(model=model, image_path="image.png")

if __name__ == "__main__":
    main(validation=True, train=True, corner_detection=True)
