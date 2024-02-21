.PHONY: aggregate_and_sort_src merge_data pull_data unpack_data

aggregate_and_sort_dest:
	./scripts/aggregate_and_sort ctu13 data dst

aggregate_and_sort_src:
	./scripts/aggregate_and_sort ctu13 data src

clean:
	rm -r ctu13
	rm -r data

create_small_data_sample:
	shuf -n 10000 data/aggregate.txt -o data/small_aggregate.txt
	sort -t ',' -k 2,2 -k 1,1 data/small_aggregate.txt > data/small_sorted_aggregate.txt

deep_clean:
	rm -r ctu13
	rm -r data
	rm ctu13.tar.gz

merge_data:
	./scripts/merge_vast_and_features ctu13/

pull_data:
	./scripts/pull_dataset

run_data_pipeline: pull_data unpack_data merge_data aggregate_and_sort_src

unpack_data:
	tar -xzvf ctu13.tar.gz
