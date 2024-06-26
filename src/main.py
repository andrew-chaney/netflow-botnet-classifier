import argparse
import os

import numpy as np
from numpy import ndarray

from models.lstm_language_model import LSTMLanguageModel
from utils import stats


def load_data(path: str) -> ndarray:
    """
    Loads the data from a CSV data file path to a Numpy array.

    :param path: path to the data file

    :returns: 2D Numpy array of strings, each row corresponding to a line in
              the file
    """
    return np.loadtxt(path, delimiter=",", dtype=str)


def split_data(data: ndarray, split: float = 0.8) -> tuple[ndarray, ndarray]:
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


def calculate_threshold(data: ndarray, model: LSTMLanguageModel) -> np.float_:
    """
    Calculates the probability threshold by taking the 95th percentile of the
    probabilities from benign data.

    :param data: Numpy array of data
    :param model: Trained LSTM Language Model

    :returns: Numpy float of the threshold
    """
    probs = []
    for ip in np.unique(data[:, 1]):
        ip_data = data[np.where(data[:, 1] == ip)]
        ip_feats = ip_data[:, 4:]
        log_prob = model.evaluate_predictions(ip_feats)
        probs.append(log_prob)
    np_probs = np.array(probs, dtype=np.float32)
    return np.percentile(np_probs[~np.isnan(np_probs)], 0.95)


def evaluate_data(data: ndarray, model: LSTMLanguageModel, thresh: float):
    """
    Evaluates provided data (intended to be used on malicious data) by
    comparing the probability of the data to the established threshold.

    :param data: Numpy array of data
    :param model: Trained LSTM Language Model
    :param thresh: probability threshold to use for comparisons
    """
    true_pos = 0
    true_neg = 0
    false_pos = 0
    false_neg = 0

    for ip in np.unique(data[:, 1]):
        ip_data = data[np.where(data[:, 1] == ip)]
        ip_feats = ip_data[:, 4:]
        log_prob = model.evaluate_predictions(ip_feats)

        if log_prob is None:
            print(f"Unable to evaluate IP Addr.: {ip}\n")
            continue

        ip_label = "benign" if ip_data[0][3] == "0" else "bot"
        derived_label = "benign" if log_prob >= thresh else "bot"

        print(f"Evaluating IP Addr. [{ip}]")
        print(f"\tThreshold: {thresh}")
        print(f"\tLog Probability: {log_prob}")
        print(f"\tTraffic from the IP is: {ip_label}")
        print(f"\tTraffic was estimated to be: {derived_label}\n")

        if ip_label == "benign":
            # Test for false positives
            if ip_label != derived_label:
                false_pos += 1
            # Test for true negatives
            else:
                true_neg += 1
        else:
            # Test for true positive
            if ip_label == derived_label:
                true_pos += 1
            # Test for false negatives
            else:
                false_neg += 1

    prec = stats.precision(true_pos, false_pos)
    rec = stats.recall(true_pos, false_neg)
    f_1 = stats.f1(true_pos, false_pos, false_neg)
    mcc = stats.mcc(true_pos, true_neg, false_pos, false_neg)

    print(f"True Positive: {true_pos}")
    print(f"True Negative: {true_neg}")
    print(f"False Positive: {false_pos}")
    print(f"False Negative: {false_neg}")

    print(f"Precision: {prec}")
    print(f"Recall: {rec}")
    print(f"F-1 Score: {f_1}")
    print(f"Matthew's Correlation Coefficient: {mcc}")


def main():
    # Process command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benign-path",
        type=str,
        required=True,
        help="Input path to the benign data file",
    )
    parser.add_argument(
        "--bot-path",
        type=str,
        required=False,
        help="Input path to the bot data file",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        required=False,
        help="Input path to the pickled tokenizer file",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        required=False,
        help="Number of times to train on the data set",
    )
    parser.add_argument(
        "--evaluate",
        default=False,
        required=False,
        action="store_true",
        help="Do you want to evaluate the bot data in addition to training?",
    )
    parser.add_argument(
        "--skip-training",
        default=False,
        required=False,
        action="store_true",
        help="Do you want to evaluate the bot data in addition to training?",
    )

    FLAGS = parser.parse_args()

    if FLAGS.evaluate and not FLAGS.bot_path:
        print("ERROR: bot path required when evaluating")
        return

    # Check the provided paths
    if not os.path.exists(FLAGS.benign_path):
        print("ERROR: benign file path provided is invalid")
        return

    if FLAGS.evaluate and not os.path.exists(FLAGS.bot_path):
        print("ERROR: bot file path provided is invalid")
        return
    else:
        bot_path = os.path.abspath(FLAGS.bot_path)
        bot_data = load_data(bot_path)

    tokenizer_path = None
    if FLAGS.tokenizer_path and not os.path.exists(FLAGS.tokenizer_path):
        print("ERROR: tokenizer file path provided is invalid")
        return
    elif FLAGS.tokenizer_path:
        tokenizer_path = os.path.abspath(FLAGS.tokenizer_path)
    else:
        print(
            "WARNING: if training in batches it is highly recommended to \
             provide a built tokenizer file to the model. \
             See scripts/build_tokenizer.py or make run_recommended_analysis"
        )

    benign_path = os.path.abspath(FLAGS.benign_path)

    # Load the benign data
    benign_data = load_data(benign_path)

    # Split the data into a training and testing set
    ben_train, ben_test = split_data(benign_data, split=0.9)

    # Set up the model and start training on benign training features
    # The features start at index 4 and go to the end of the array
    model = LSTMLanguageModel(ben_train[:, 4:], tokenizer_path, FLAGS.epochs)

    if not FLAGS.skip_training:
        model.train()
    else:
        if not FLAGS.tokenizer_path:
            print("WARNING: tokenizer path is required to skip training")
            return
        model.model_trained = True

    # Evaluate the predictions that the model makes
    if FLAGS.evaluate:
        threshold = float(calculate_threshold(ben_test, model))
        print(f"Calculated threshold is: {threshold}")
        evaluate_data(np.concatenate((ben_test, bot_data)), model, threshold)


if __name__ == "__main__":
    main()
