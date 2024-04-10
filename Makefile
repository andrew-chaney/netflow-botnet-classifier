.PHONY: aggregate_and_sort_src merge_data pull_data unpack_data

aggregate_and_sort_dest:
	./scripts/aggregate_and_sort ctu13 data dst

aggregate_and_sort_src:
	./scripts/aggregate_and_sort ctu13 data src

clean:
	rm -r ctu13
	rm -r data
	rm -r src/**/__pycache__

create_small_data_sample:
	shuf -n 10000 data/aggregate.txt -o data/small_aggregate.txt
	sort -t ',' -k 2,2 -k 1,1 data/small_aggregate.txt > data/small_sorted_aggregate.txt
	awk -F ',' '{ if ($$4==0) {print $$0}}' data/small_sorted_aggregate.txt > data/small_benign_traffic.txt
	awk -F ',' '{ if ($$4==1) {print $$0}}' data/small_sorted_aggregate.txt > data/small_bot_traffic.txt

deep_clean:
	rm -r ctu13
	rm -r data
	rm -r src/**/__pycache__
	rm ctu13.tar.gz lstm_model_epochs_*.keras

merge_data:
	./scripts/merge_vast_and_features ctu13/

pull_data:
	./scripts/pull_dataset

run_data_pipeline: pull_data unpack_data merge_data

run_full_lang_analysis:
	python src/main.py --benign-path data/sorted_benign_traffic.txt --bot-path data/sorted_bot_traffic.txt

run_recommended_analysis:
	./scripts/recommended_analysis

run_micro_recommended_analysis:
	./scripts/micro_recommended_analysis

unpack_data:
	tar -xzvf ctu13.tar.gz
