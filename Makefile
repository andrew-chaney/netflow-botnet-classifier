clean:
	rm -r ctu13
	rm ctu13.tar.gz

pull_data:
	./scripts/pull_dataset

unpack_data:
	tar -xzvf ctu13.tar.gz
