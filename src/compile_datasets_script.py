from datafetcher import compile_fred_training_dataset
from datafetcher import compile_congress_training_dataset

# Script parameters
compile_congress_dataset = False
compile_fred_dataset = True
retry_congress_errors = False

if __name__ == '__main__':
    if compile_congress_dataset:  # Download and compile Congress dataset
        # Assign script variables here
        kwargs = {
            "n_max_results": 250
        }
        compile_congress_training_dataset.script_to_run(**kwargs)

    if compile_fred_dataset:  # Download and compile FRED dataset
        # Assign script variables here
        kwargs = {
            "min_popularity": 85
        }
        compile_fred_training_dataset.script_to_run(**kwargs)

    if retry_congress_errors:
        compile_congress_training_dataset.retry_errors()
