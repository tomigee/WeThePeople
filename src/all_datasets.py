from .datafetcher import fred_data
from .datafetcher import congress_data


def get(compile_congress_dataset,
        compile_fred_dataset,
        retry_congress_errors,
        congress_args={},
        fred_args={},
        retry_args={}):

    if compile_congress_dataset:  # Download and compile Congress dataset
        congress_data.get(**congress_args)
    if compile_fred_dataset:  # Download and compile FRED dataset
        fred_data.get(**fred_args)
    if retry_congress_errors:  # Retry failed congress downloads
        congress_data.retry_errors(**retry_args)
