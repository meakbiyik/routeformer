# How is test data (e.g., under tests/io/data) generated?

The script `scripts/prepare-test-files.sh` is used to generate the test data.

We use the real videos to test the video reader properly. The test data is not used for training or evaluating the model, it is just for testing the Python code.
