from os import path
import pickle
import matplotlib.pyplot as plt


def run_analysis():
    k_data = [[] for x in range(4)]
    if path.exists("k_data.pickle"):
        k_data = pickle.load(open("k_data.pickle", "rb"))
    training_losses, testing_losses, training_accuracies, testing_accuracies, n = k_data

    num_runs = len(k_data)
    print(f"Loading data from {num_runs} runs")

    plotter("accuracy", training_accuracies, testing_accuracies)
    plotter("loss", training_losses, testing_losses)


def plotter(metric, n, train_data, test_data):
    x = range(0, n)

    if metric == "accuracy":
        symbol = "%"
        line = "-"
    else:
        symbol = "MSE"
        line = "--"

    plt.scatter(x, train_data, color='b', marker='^')
    plt.plot(x, train_data, f'b{line}')

    plt.scatter(x, test_data, color='r', marker='s')
    plt.plot(x, test_data, f'r{line}')

    plt.title(f"training and testing {metric}")
    plt.xlabel("epochs")
    plt.ylabel(f"{metric} ({symbol})")

    plt.show()
    plt.clf()


if __name__ == "__main__":
    run_analysis()
