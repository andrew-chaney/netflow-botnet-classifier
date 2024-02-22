import argparse
import os

import numpy as np

from models.lstm_language_model import LSTMLanguageModel


def load_data(path: str) -> np.ndarray:
    """
    Loads the data from a CSV data file path to a Numpy array.

    :param path: path to the data file

    :returns: 2D Numpy array of strings, each row corresponding to a line in
              the file
    """
    return np.loadtxt(path, delimiter=',', dtype=str)


def split_data(data: np.ndarray, split: float=0.8) -> (np.ndarray, np.ndarray):
    """
    Splits data into training and testing sets based on unique IPs.

    :param data: Numpy array of data
    :param split: percent of data to train on (default: 80%)

    :returns: tuple of Numpy arrays (training, testing)
    """
    # Find all unique IPs
    unique_ips = np.unique(data[:, 1])
    # Get the training size and IPs to use for training
    train_size = int((unique_ips.size * split) // 1)
    train_ips = np.random.choice(unique_ips, size=train_size, replace=False)
    # Create a mask and use the mask to build the training and testing sets
    mask = np.in1d(data[:, 1], train_ips)
    training_set = data[mask]
    testing_set = data[~mask]

    return (training_set, testing_set)


def main():
    # Process command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benign-path",
        type=str,
        required=True,
        help="Input path to the sorted, benign data file",
    )
    parser.add_argument(
        "--bot-path",
        type=str,
        required=True,
        help="Input path to the sorted, bot data file",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        required=False,
        help="Number of times to train on the data set",
    )

    FLAGS = parser.parse_args()

    # Check the provided paths
    if not os.path.exists(FLAGS.benign_path):
        print("ERROR: benign file path provided is invalid")
        return

    if not os.path.exists(FLAGS.bot_path):
        print("ERROR: bot file path provided is invalid")
        return

    benign_path = os.path.abspath(FLAGS.benign_path)
    bot_path = os.path.abspath(FLAGS.bot_path)

    # Load the data
    benign_data = load_data(benign_path)
    bot_data = load_data(bot_path)

    # Split the data into a training and testing set
    ben_train, ben_test = split_data(benign_data)
    bot_train, bot_test = split_data(bot_data)

    # Discard the timestamps, source ips, and destination ips
    ben_train = np.delete(ben_train, np.s_[:3], axis=1)
    ben_test = np.delete(ben_test, np.s_[:3], axis=1)
    bot_train = np.delete(bot_train, np.s_[:3], axis=1)
    bot_test = np.delete(bot_test, np.s_[:3], axis=1)

    # Split off the labels from the features
    ben_train_label = ben_train[:, 0]
    ben_train_feat = ben_train[:, 1:]
    ben_test_label = ben_test[:, 0]
    ben_test_feat = ben_test[:, 1:]
    bot_train_label = bot_train[:, 0]
    bot_train_feat = bot_train[:, 1:]
    bot_test_label = bot_test[:, 0]
    bot_test_feat = bot_test[:, 1:]

    # Set up the model and start training on benign training data.
    model = LSTMLanguageModel(ben_train_feat, epochs=FLAGS.epochs)
    model.train()
    # Evaluate the predictions that the model makes
    benign_log_prob = model.evaluate_predictions(ben_test_feat)
    bot_log_prob = model.evaluate_predictions(bot_test_feat)
    print(f"Avg. Log Probability for Benign Testing: {benign_log_prob}")
    print(f"Avg. Log Probability for Bot Testing:    {bot_log_prob}")


if __name__ == "__main__":
    main()
