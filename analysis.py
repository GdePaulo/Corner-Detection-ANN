import os
from os import path
import pickle

def runAnalysis():
    k_data = [[] for x in range(4)]
    if  path.exists("k_data.pickle"):
        k_data = pickle.load(open("k_data.pickle", "rb"))
    training_losses, testing_losses, training_accuracies, testing_accuracies = k_data
    
    print(f"Loading data from {len(k_data)} runs")
        
if __name__ == "__main__":
    runAnalysis()
