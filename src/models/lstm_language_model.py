from tqdm import tqdm

from keras import Input
from keras.layers import Dense, Embedding, LSTM
from keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import numpy as np
from numpy import ndarray


class LSTMLanguageModel:
    """
    LSTM Model that works to predict the next feature from a set of netflow
    data features.

    :param features: 2D Numpy array of Netflow features data
    :param epochs: Number of times for model to train on the provided dataset
    """

    def __init__(self, features: ndarray, epochs: int = 100):
        # Persist training data
        self.X = features
        self.epochs = epochs
        # Flags for defining LSTM state
        self.data_ready = False
        self.model_ready = False
        self.model_trained = False

    def train(self, verbose: bool = True) -> None:
        """
        Trains the model on the data, preparing the data and model if needed.

        :param verbose: Display output as the model trains
        """
        if self.model_trained:
            return

        if not self.data_ready or not self.model_ready:
            self.__prep_training_data__()
            self.__activate_layers__()

        v = 1 if verbose else 0
        self.model.fit(self.X, self.y, epochs=self.epochs, verbose=v)
        self.model_trained = True

    def evaluate(self, data: ndarray) -> None:
        """
        Run the model's evaluation function given some testing data.

        :param data: 2D Numpy array of Netflow features data
        """
        if not self.model_trained:
            self.train()

        test_X, test_y = self.__prep_testing_data__(data)
        self.model.evaluate(test_X, test_y)

    def evaluate_predictions(self, data: ndarray) -> np.float_:
        """
        Given some testing data, find the probability of accurate predictions
        throughout a Netflow's sequence of features.

        :param data: 2D Numpy array of Netflow features data

        :returns: Numpy Float of the average log probability for the sequences
                  based on the model's understanding
        """
        if not self.model_trained:
            self.train()

        test_X, test_y = self.__prep_testing_data__(data)
        probabilities = []
        predictions = self.model.predict(test_X)
        for y, probs in zip(test_y, predictions):
            probabilities.append(np.log(probs[np.where(y == 1)[0][0]]))
        return np.average(probabilities)

    def __activate_layers__(self) -> None:
        """
        Set up the LSTM Model and all of its layers for testing.
        """
        if self.model_ready:
            return

        if not self.data_ready:
            self.__prep_training_data__()

        # Instantiate the model
        self.model = Sequential()
        # Layer 0 - Input Layer
        self.model.add(Input(shape=(self.X.shape[1],)))
        # Layer 1 - Embedding Layer
        self.model.add(Embedding(self.total_words, 100))
        # Layer 2 - LSTM Layer
        self.model.add(LSTM(150, activation="tanh"))
        # Layer 3 - SoftMax (Output) Layer
        self.model.add(Dense(self.total_words, activation="softmax"))
        self.model.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        self.model_ready = True

    def __tokenize__(self, data: ndarray) -> None:
        """
        Fit the Tokenizer to the data sequences.

        :param data: Numpy array of data sequence in the form of strings
        """
        self.tokenizer = Tokenizer(filters='!"#$%&()*,/:;<=>?@[\\]^_`{|}~\t\n')
        for seq in tqdm(data, desc="Fitting Tokenizer"):
            self.tokenizer.fit_on_texts([seq])
        self.total_words = len(self.tokenizer.word_index) + 1

    def __sequence__(self, data: ndarray) -> None:
        """
        Generate padded, tokenized sequences to train on from the data. Saves
        generated sequences and information to the object to be called later.

        :param data: 1D Numpy array of Netflow feature data strings
        """
        self.sequences = []
        # Generate tokenized sequences
        for seq in tqdm(data, desc="Tokenizing Sequences"):
            tokens = self.tokenizer.texts_to_sequences([seq])[0]
            for i in range(1, len(tokens)):
                self.sequences.append(tokens[: i + 1])
        # Pad the sequences to the same length
        self.max_len = max([len(seq) for seq in self.sequences])
        self.sequences = np.array(
            pad_sequences(
                self.sequences,
                maxlen=self.max_len,
                padding="pre",
            )
        )

    def __flatten_to_strs__(self, arr: ndarray, delim: str = " ") -> ndarray:
        """
        Takes a 2D Numpy array, flattens each sub-array to a string joined by
        a specified delimiter, and returns the resulting 1D array of strings.

        :param arr: 2D Numpy array of data
        :param delim: delimiter for separating strings

        :returns: 1D Numpy array of strings
        """
        return np.fromiter(map(lambda x: delim.join(x), arr), dtype=object)

    def __prep_training_data__(self) -> None:
        """
        Run the data preparation pipeline on the provided Netflow features
        data.
        """
        if self.data_ready:
            return

        self.X = self.__flatten_to_strs__(self.X)
        self.__tokenize__(self.X)
        self.__sequence__(self.X)
        self.X = self.sequences[:, :-1]
        self.y = np.array(
            to_categorical(
                self.sequences[:, -1],
                num_classes=self.total_words,
            )
        )
        self.data_ready = True

    def __prep_testing_data__(self, data: ndarray) -> tuple[ndarray, ndarray]:
        """
        Run the data preparation pipeline on the provided Netflow features
        testing data.

        :param data: 2D Numpy array of Netflow features data

        :returns: Tuple of Numpy arrays in the form of (X, y)
        """
        data = self.__flatten_to_strs__(data)
        sequences = []
        for d in data:
            # Tokenize the data point and break into sequences to predict on
            tokens = self.tokenizer.texts_to_sequences([d])[0]
            sequences.extend([tokens[: i + 1] for i in range(1, len(tokens))])
        sequences = pad_sequences(
            sequences,
            maxlen=self.max_len,
            padding="pre",
        )
        X = sequences[:, :-1]
        y = np.array(
            to_categorical(
                sequences[:, -1],
                num_classes=self.total_words,
            )
        )
        return (X, y)
