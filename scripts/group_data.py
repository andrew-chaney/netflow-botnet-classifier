import argparse
import os
import time
from tqdm import tqdm

import numpy as np
import pandas as pd

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


def write_df_to_csv(df: pd.DataFrame, ip: str, output_path: str) -> None:
    file_name = "{}.txt".format(ip.replace('.', ''))
    if ':' in ip:
        file_name = file_name.replace(':', '')

    output_file = os.path.join(output_path, file_name)
    with open(output_file, 'a') as file:
        np.savetxt(file, df.to_numpy(), delimiter=',', fmt="%s")


@function_timer
def org_files_by_ip_addr(paths: [str], output_path: str) -> None:
    for path in tqdm(paths, desc="Organizing Data Files By IP Addresses"):
        df = pd.read_csv(path, names=FEATURE_COLS)
        for ip in df.src_ip.unique():
            ip_df = df.loc[df.src_ip == ip]
            write_df_to_csv(ip_df, ip, output_path)


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

    org_files_by_ip_addr(data_files, output_path)


if __name__ == "__main__":
    main()
