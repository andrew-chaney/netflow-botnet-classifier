import numpy as np

from src.main import load_data, split_data
from src.models.lstm_language_model import LSTMLanguageModel

TEST_DATA_PATH = "./test/resources/test_data.txt"
EXPECTED_DATA_PATH = "./test/resources/expected_prepared_data.txt"


def test_data_prepartion():
    # Get the expected prepared data
    expected_data = np.loadtxt(
        EXPECTED_DATA_PATH, delimiter=",", dtype=np.int32)

    # Get and prep the data for model
    data = load_data(TEST_DATA_PATH)
    # Seed numpy so that we know the expected data
    np.random.seed(1551)
    train, _ = split_data(data)

    # Load the data into the model and prep the training data
    model = LSTMLanguageModel(train[:, 4:])
    model.__prep_training_data__()

    # Evaluate the prepared data
    assert model.X.shape == (567, 27)
    assert model.X.shape == expected_data.shape
    assert np.array_equal(model.X, expected_data)
