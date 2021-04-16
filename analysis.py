
def runAnalysis():
    k_data = [[] for x in range(4)]
    if  path.exists("k_data.pickle"):
        k_data = pickle.load(open("k_data.pickle", "rb"))
    training_losses, testing_losses, training_accuracies, testing_accuracies = k_data
        
if __name__ == "__main__":
    runAnalysis()
