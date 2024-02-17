.PHONY: aggregate_and_sort_src merge_data pull_data unpack_data

aggregate_and_sort_dest:
	./scripts/aggregate_and_sort ctu13 dst

aggregate_and_sort_src:
	./scripts/aggregate_and_sort ctu13 src

clean:
	rm -r ctu13
	rm aggregate.txt
	rm sorted_aggregate.txt

deep_clean:
	rm -r ctu13
	rm ctu13.tar.gz
	rm aggregate.txt
	rm sorted_aggregate.txt

merge_data:
	./scripts/merge_vast_and_features ctu13/

pull_data:
	./scripts/pull_dataset

run_data_pipeline: pull_data unpack_data merge_data aggregate_and_sort_src

unpack_data:
	tar -xzvf ctu13.tar.gz
