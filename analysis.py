from os import path
import pickle
import matplotlib.pyplot as plt

analysis_types = ["epochs", "data_set_sizes"]
analysis_type = analysis_types[1]
experiment_path = "experiments/feedforward_validation/"
experiment_file = "k_data_validation.pickle"


def run_analysis():
    if path.exists(experiment_path + experiment_file):
        k_data = pickle.load(open(experiment_path + experiment_file, "rb"))

    training_losses, testing_losses, training_accuracies, testing_accuracies, x_data = k_data
    plotter("accuracy", x_data, training_accuracies, testing_accuracies)
    plotter("loss", x_data, training_losses, testing_losses)


def plotter(metric, x, train_data, test_data):
    x_lbl = ""

    # The x-axis depends on the analysis type that is going to be displayed
    if analysis_type == analysis_types[0]:
        x_lbl = "epochs"
    elif analysis_type == analysis_types[1]:
        x_lbl = "data set sizes"

    # The y-axis depends on the metric that is going to be displayed
    if metric == "accuracy":
        symbol = "%"
        line = "-"
    else:
        symbol = "BCE"
        line = "--"

    plt.scatter(x, train_data, color='b', marker='^')
    plt.plot(x, train_data, f'b{line}', label="train")

    plt.scatter(x, test_data, color='r', marker='s')
    plt.plot(x, test_data, f'r{line}', label="test")

    if len(x) < 10:
        plt.xticks(x)

    plt.title(f"training and testing {metric} for ANN")
    plt.xlabel(x_lbl)
    plt.ylabel(f"{metric} ({symbol})")
    plt.legend()
    
    # plt.show()
    plt.savefig(experiment_path + metric + ".png")
    plt.clf()


if __name__ == "__main__":
    run_analysis()
