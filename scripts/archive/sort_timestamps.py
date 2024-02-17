import argparse
import os
from tqdm import tqdm

import numpy as np

from utils import function_timer


def sort_file(path: str) -> None:
    data = np.loadtxt(path, delimiter=',', dtype=str)

    # If there is only a single line in the file, skip sorting
    if len(data.shape) < 2:
        return

    sorted_data = data[data[:, 0].argsort()]
    np.savetxt(path, sorted_data, delimiter=',', fmt="%s")


@function_timer
def process_files(input_dir: str):
    file_lst = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            file_lst.append(os.path.join(root, file))

    [sort_file(file) for file in tqdm(file_lst, desc="Sorting Files")]


def main():
    # Process command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-path",
        type=str,
        required=True,
        help="The path to cleaned data to sort.",
    )

    FLAGS = parser.parse_args()

    # Convert path arguments to absolute file paths
    input_path = os.path.abspath(FLAGS.input_path)

    process_files(input_path)


if __name__ == "__main__":
    main()
