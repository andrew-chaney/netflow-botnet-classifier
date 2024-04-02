import os

import numpy as np

from src.main import load_data, split_data

TEST_DATA_PATH = "./test/resources/test_data.txt"


def test_loading_data():
    data = load_data(TEST_DATA_PATH)
    assert data.shape == (26, 32)


def test_splitting_data():
    data = load_data(TEST_DATA_PATH)
    train, test = split_data(data)

    uniq_ips = np.unique(data[:, 1])
    train_size = int((uniq_ips.size * 0.8) // 1)

    # Training data should be 80% of the inputted data unique IPs
    assert np.unique(train[:, 1]).shape[0] == train_size
    # Testing data should be 20% of the inputted data unique IPs
    assert np.unique(test[:, 1]).shape[0] == (uniq_ips.size - train_size)
    # Overall sizes should match
    assert (train.shape[0] + test.shape[0]) == data.shape[0]
