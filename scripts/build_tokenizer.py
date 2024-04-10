import os
import pickle
import re
from tqdm import tqdm

import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer

cur = None
tokenizer = Tokenizer(filters='!"#$%&()*,/:;<=>?@[\\]^_`{|}~\t\n')


def flatten_to_strs(arr, delim=" "):
    return np.fromiter(map(lambda x: delim.join(x), arr), dtype=object)


def tokenize(data, path):
    tokenizer = Tokenizer(filters='!"#$%&()*,/:;<=>?@[\\]^_`{|}~\t\n')
    for seq in tqdm(data, desc=path):
        tokenizer.fit_on_texts([seq])


def process_file(path):
    data = np.loadtxt(path, delimiter=",", dtype=str)
    data = flatten_to_strs(data)
    tokenize(data, path)


def pickle_tokenizer():
    print(f"Pickling tokenizer: {cur}")
    with open("data/tokenizer-{}.pickle".format(cur), "wb") as file:
        pickle.dump(tokenizer, file, protocol=pickle.HIGHEST_PROTOCOL)


def get_id_from_path(path):
    split_idx = -1
    if "benign" in path:
        split_idx = -2
    path_str = path.split("/")[split_idx]
    return int(re.search(r"\d{1,2}", path_str).group())


def get_current_tokenizer():
    last = 0
    for _, _, files in os.walk("data"):
        for file in files:
            if "tokenizer" in file:
                file_num = get_id_from_path(file)
                last = max(last, file_num)
    return last + 1


def main():
    print(
        "\nNote: If this program stops runnning before it completes then simply rerun it. It has the ability to pick up where it left off.\n"
    )

    cur = get_current_tokenizer()
    paths = []
    for root, _, files in os.walk("ctu13"):
        for file in files:
            if "benign" in file:
                paths.append(os.path.join(root, file))

    for path in tqdm(
        sorted(paths, key=get_id_from_path), desc="Tokenizing Input Files"
    ):
        if get_id_from_path(path) < cur:
            continue
        process_file(path)
        pickle_tokenizer()
        cur += 1


if __name__ == "__main__":
    main()
