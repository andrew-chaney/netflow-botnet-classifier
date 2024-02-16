clean:
	rm -r ctu13
	rm -r out
	rm -r prog
	rm nohup.out

deep_clean:
	rm -r ctu13
	rm -r out
	rm -r prog
	rm nohup.out
	rm ctu13.tar.gz

group_data:
	nohup ./scripts/group_with_bash ctu13 out src &
	# python scripts/group_data.py --input-path ctu13 --output-path out

merge_data:
	./scripts/merge_vast_and_features ctu13/

pull_data:
	./scripts/pull_dataset

unpack_data:
	tar -xzvf ctu13.tar.gz
