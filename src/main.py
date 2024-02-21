import argparse
import os

import numpy as np


def load_data(path: str) -> np.ndarray:
    return np.loadtxt(path, delimiter=',', dtype=str)


def split_data(data: np.ndarray, split: float=0.8) -> (np.ndarray, np.ndarray):
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
        "--input-path",
        type=str,
        required=True,
        help="Input path to the sorted and aggregated data file",
    )

    FLAGS = parser.parse_args()

    # Check the provided path
    if not os.path.exists(FLAGS.input_path):
        print("ERROR: path provided is invalid")
        return

    input_path = os.path.abspath(FLAGS.input_path)

    # Load the data
    data = load_data(input_path)

    # Split the data into a training and testing set
    train, test = split_data(data)

    # Discard the timestamps, source ips, and destination ips
    train = np.delete(train, np.s_[:3], axis=1)
    test = np.delete(test, np.s_[:3], axis=1)

    # Split off the labels from the features
    train_label = train[:, 0]
    train_feat = train[:, 1:]
    test_label = test[:, 0]
    test_feat = test[:, 1:]


if __name__ == "__main__":
    main()
