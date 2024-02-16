import argparse
from concurrent.futures import ProcessPoolExecutor
import os
import time
from tqdm.auto import tqdm

import numpy as np

FEATURE_COLS = [
    "timestamp",  # From merge_vast_and_features script output
    "src_ip",     # From merge_vast_and_features script output
    "dest_ip",    # From merge_vast_and_features script output
    "bot_flag",   # 0 if not a bot, 1 if a bot
    "feature0",   # FOREACH VerticesBySrc ave(SrcTotalBytes,10000,2)
    "feature1",   # FOREACH VerticesBySrc var(SrcTotalBytes,10000,2)
    "feature2",   # FOREACH VerticesBySrc ave(DestTotalBytes,10000,2)
    "feature3",   # FOREACH VerticesBySrc var(DestTotalBytes,10000,2)
    "feature4",   # FOREACH VerticesBySrc ave(DurationSeconds,10000,2)
    "feature5",   # FOREACH VerticesBySrc var(DurationSeconds,10000,2)
    "feature6",   # FOREACH VerticesBySrc ave(SrcPayloadBytes,10000,2)
    "feature7",   # FOREACH VerticesBySrc var(SrcPayloadBytes,10000,2)
    "feature8",   # FOREACH VerticesBySrc ave(DestPayloadBytes,10000,2)
    "feature9",   # FOREACH VerticesBySrc var(DestPayloadBytes,10000,2)
    "feature10",  # FOREACH VerticesBySrc ave(FirstSeenSrcPacketCount,10000,2)
    "feature11",  # FOREACH VerticesBySrc var(FirstSeenSrcPacketCount,10000,2)
    "feature12",  # FOREACH VerticesBySrc ave(FirstSeenDestPacketCount,10000,2)
    "feature13",  # FOREACH VerticesBySrc var(FirstSeenDestPacketCount,10000,2)
    "feature14",  # FOREACH VerticesByDest ave(SrcTotalBytes,10000,2)
    "feature15",  # FOREACH VerticesByDest var(SrcTotalBytes,10000,2)
    "feature16",  # FOREACH VerticesByDest ave(DestTotalBytes,10000,2)
    "feature17",  # FOREACH VerticesByDest var(DestTotalBytes,10000,2)
    "feature18",  # FOREACH VerticesByDest ave(DurationSeconds,10000,2)
    "feature19",  # FOREACH VerticesByDest var(DurationSeconds,10000,2)
    "feature20",  # FOREACH VerticesByDest ave(SrcPayloadBytes,10000,2)
    "feature21",  # FOREACH VerticesByDest var(SrcPayloadBytes,10000,2)
    "feature22",  # FOREACH VerticesByDest ave(DestPayloadBytes,10000,2)
    "feature23",  # FOREACH VerticesByDest var(DestPayloadBytes,10000,2)
    "feature24",  # FOREACH VerticesByDest ave(FirstSeenSrcPacketCount,10000,2)
    "feature25",  # FOREACH VerticesByDest var(FirstSeenSrcPacketCount,10000,2)
    "feature26",  # FOREACH VerticesByDest ave(FirstSeenDestPacketCount,10000,2)
    "feature27",  # FOREACH VerticesByDest var(FirstSeenDestPacketCount,10000,2)
]


def function_timer(func):
    def decorator(*args, **kwargs):
        t1 = time.time()
        result = func(*args, **kwargs)
        t2 = time.time()
        print(f"{func.__name__}() executed in {(t2-t1):.6f}s")
        return result
    return decorator


@function_timer
def gather_merged_data_files(in_path: str) -> [str]:
    output = []
    for root, _, files in os.walk(in_path):
        for file in files:
            if file == "merged.txt":
                output.append(os.path.join(root, file))
    return output


def write_arr_to_csv(arr: np.array, ip: str, output_path: str) -> None:
    translation_table = str.maketrans({'.': '', ':': ''})
    # We'll make three levels to the output directory due to the amount
    # of output.
    nested_dir = os.path.join(
        output_path,
        ip.translate(translation_table)[:3],
    )

    if not os.path.exists(nested_dir):
        os.mkdir(nested_dir)

    output_dir = os.path.join(
        nested_dir,
        ip.translate(translation_table)[:6],
    )

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    output_file = os.path.join(
        output_dir,
        "{}_grouped.txt".format(ip),
    )
    with open(output_file, 'a') as file:
        np.savetxt(file, arr, delimiter=',', fmt="%s")


def process_file(path: str, output_path: str) -> None:
    src_ip_idx = FEATURE_COLS.index("src_ip")
    data = np.loadtxt(path, delimiter=',', dtype=str)
    for ip in np.unique(data[:, src_ip_idx]):
        ip_data = data[data[:, src_ip_idx] == ip, :]
        write_arr_to_csv(ip_data, ip, output_path)


@function_timer
def org_files_by_ip_addr(paths: [str], output_path: str, workers=None) -> None:
    # Process input files in parallel
    with tqdm(total=len(paths), desc="Organizing Data Files By IP Addresses") as progress:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for p in paths:
                future = executor.submit(process_file, p, output_path)
                future.add_done_callback(lambda x: progress.update())


@function_timer
def verify_output(input_files: [str], output_dir: str) -> None:
    print("Verifying output line count...")
    input_line_count = 0
    for in_file in input_files:
        with open(in_file, 'r') as file:
            for line in file:
                input_line_count += 1

    output_line_count = 0
    for root, _, files in os.walk(output_dir):
        for f in files:
            with open(os.path.join(root, f), 'r') as file:
                for line in file:
                    output_line_count += 1

    if input_line_count == output_line_count:
        print("Output matches input.")
    else:
        print("Output does NOT match input.")


def main():
    tqdm.pandas()

    # Process command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-path",
        type=str,
        required=True,
        help="The path to the ctu13 dataset.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Directory to output the organized data to.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        required=False,
        default=None,
        help="Number of workers to use.",
    )

    FLAGS = parser.parse_args()

    # Convert path arguments to absolute file paths
    input_path = os.path.abspath(FLAGS.input_path)
    output_path = os.path.abspath(FLAGS.output_path)

    # If the specified output directory doesn't exist, create it
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    # Gather all merged data files
    data_files = gather_merged_data_files(input_path)

    if len(data_files) == 0:
        print("ERROR: no merged data files found")
        return

    print(f"{len(data_files)} data file(s) found.")

    org_files_by_ip_addr(data_files, output_path, FLAGS.workers)
    verify_output(data_files, output_path)


if __name__ == "__main__":
    main()
