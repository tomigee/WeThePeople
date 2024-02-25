from datafetcher import compile_fred_training_dataset
from datafetcher import compile_congress_training_dataset

# Script parameters
compile_congress_dataset = False
compile_fred_dataset = False

if compile_congress_dataset:
    # Download and compile Congress dataset
    if __name__ == '__main__':
        compile_congress_training_dataset.script_to_run()

if compile_fred_dataset:
    # Download and compile FRED dataset
    if __name__ == '__main__':
        compile_fred_training_dataset.script_to_run()
