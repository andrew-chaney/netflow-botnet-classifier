# Netflow Botnet Classifier
This is a project that is seeking to be able to use machine learning to classify network traffic as either genuine traffic or traffic generated by a botnet.

## Commands to Run
To recreate the project, you can run the following commands.
1. `make run_data_pipeline`: This command will run the entire pipeline from pulling and unpacking the data to merging and aggregating it all.

## Directory and File Overview
- `src/`: source-code for the machine learning models and data analyis
- `scripts/`: scripts for pulling and preparing the data
- `ctu13/`:
    - `**/captureAsVast.txt`: vast capture of netflow traffic for a specific scenario
    - `**/features.txt`: netflow data broken down by the following features
	    - Bot Flag: 0 if not a bot, 1 if a bot
	    - Source Total Bytes Avg. (from analysis of traffic source)
	    - Source Total Bytes Var. (from analysis of traffic source)
	    - Destination Total Bytes Avg. (from analysis of traffic source)
	    - Destination Total Bytes Var. (from analysis of traffic source)
	    - Duration in Seconds Avg. (from analysis of traffic source)
	    - Duration in Seconds Var. (from analysis of traffic source)
	    - Source Payload Bytes Avg. (from analysis of traffic source)
	    - Source Payload Bytes Var. (from analysis of traffic source)
	    - Destination Payload Bytes Avg. (from analysis of traffic source)
	    - Destination Payload Bytes Var. (from analysis of traffic source)
	    - First Seen Source Packet Count Avg. (from analysis of traffic destination)
	    - First Seen Source Packet Count Var. (from analysis of traffic destination)
	    - First Seen Destination Packet Count Avg. (from analysis of traffic destination)
	    - First Seen Destination Packet Count Var. (from analysis of traffic destination)
	    - Source Total Bytes Avg. (from analysis of traffic destination)
	    - Source Total Bytes Var. (from analysis of traffic destination)
	    - Destination Total Bytes Avg. (from analysis of traffic destination)
	    - Destination Total Bytes Var. (from analysis of traffic destination)
	    - Duration in Seconds Avg. (from analysis of traffic destination)
	    - Duration in Seconds Var. (from analysis of traffic destination)
	    - Source Payload Bytes Avg. (from analysis of traffic destination)
	    - Source Payload Bytes Var. (from analysis of traffic destination)
	    - Destination Payload Bytes Avg. (from analysis of traffic destination)
	    - Destination Payload Bytes Var. (from analysis of traffic destination)
	    - First Seen Source Packet Count Avg. (from analysis of traffic destination)
	    - First Seen Source Packet Count Var. (from analysis of traffic destination)
	    - First Seen Destination Packet Count Avg. (from analysis of traffic destination)
	    - First Seen Destination Packet Count Var. (from analysis of traffic destination)
    - `**/merged.txt`: netflow feature data with three additional rows prepended to the feature data
        - Timestamp of the traffic
        - Source IP Address
        - Destination IP Address
    - `**/timestamps.txt`: timestamps pulled from the scenario's `captureAsVast.txt`
    - `**/*_ips.txt`: IP addresses (either source [src] or destination [dst]) pulled from the scenario's `captureAsVast.txt`
