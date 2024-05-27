from .datafetcher import fred_data
from .datafetcher import congress_data


def get(
    compile_congress_dataset: bool,
    compile_fred_dataset: bool,
    retry_congress_errors: bool,
    congress_args: dict[str,] = {},
    fred_args: dict[str,] = {},
    retry_args: dict[str,] = {},
) -> None:
    """Download and compile the Congress.gov bill text dataset and FRED dataset.

    Args:
        compile_congress_dataset (bool): Get Congress.gov dataset.
        compile_fred_dataset (bool): Get FRED dataset.
        retry_congress_errors (bool): Retry failed downloads encountered during Congress.gov dataset download.
        congress_args (dict[str,], optional): Configuration arguments passed to `congress_data`. Defaults to {}.
        fred_args (dict[str,], optional): Configuration arguments passed to `fred_data`. Defaults to {}.
        retry_args (dict[str,], optional): Configuration arguments passed to retry method of `congress_data`. Defaults to {}.
    """  # noqa: E501

    if compile_congress_dataset:  # Download and compile Congress dataset
        congress_data.get(**congress_args)
    if compile_fred_dataset:  # Download and compile FRED dataset
        fred_data.get(**fred_args)
    if retry_congress_errors:  # Retry failed congress downloads
        congress_data.retry_errors(**retry_args)
