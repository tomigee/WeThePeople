# This file contains a script that compiles:
# 1. Time series of the 10 most popular economic indicators from FRED
#
# fred/tags/series is my answer
#
# Steps:
# 1. Get all tag names from fred/tags, compile into a list
# 2. Use the compiled tag names with fred/tags/series to get all series, and sort by popularity
# 3. Download the N most popular ones

import os
from os import path
import traceback as tb
import re
from collections.abc import Generator
from datetime import datetime, timedelta

from tqdm import tqdm

from .fred_data import (
    FredSeries,
    FredTag,
    FredResponse,
)

from .fred.fred.core import Fred
from .utils import (
    json_to_csv,
    truncate_filepath,
    process_filename,
    compose_error_entry,
    write_error_files,
    write_files,
    load_json_file,
)


class FredDatafetcher:
    def __init__(self):
        self.client = Fred()
        self.max_items_per_request = 1000
        self.test_limit = 2
        self.throttle = True
        self.fred_series_info_path = "datasets/fred_series_info.json"
        self.error_log = []
        self.error_log_filepath = "error logs/error_log.json"
        self.training_data_path = "datasets/training data"
        self.fred_data_path = "datasets/FredData"
        self.min_popularity = 85

    def get_all_tag_generators_from_fred(
        self,
    ) -> list[Generator[FredTag, None, None]] | None:
        """Get all tag names from FRED.

        Returns:
            List of tag names.
        """
        OFFSET = 0
        all_tag_generators = []
        successful_probe = self.probe_endpoint(self.client.tags)
        if successful_probe:
            while OFFSET < successful_probe.metadata["count"]:
                temp = self._api_query(
                    self.client.tags,
                    limit=self.max_items_per_request,
                    offset=OFFSET,
                    throttle=True,
                )
                if temp is None:
                    OFFSET += self.max_items_per_request
                    continue
                all_tag_generators.append(temp.parsed_payload)
                OFFSET += self.max_items_per_request
        else:
            return None
        return all_tag_generators

    def get_error_log(self, endpoint: str, path: str = "") -> list[dict[str, str]]:
        if endpoint == "tags" and path == "":
            return self.tags_request_error_log
        elif endpoint == "tags" and path == "series":
            return self.tags_data_error_log
        elif endpoint == "observations":
            return self.observations_request_error_log
        else:
            return self.observations_data_error_log

    def probe_endpoint(self, function, **kwargs) -> FredResponse:
        kwargs["throttle"] = self.throttle
        kwargs["limit"] = self.test_limit
        return self._api_query(function, **kwargs)

    def _api_query(self, api_function, **kwargs):
        try:
            response = FredResponse(api_function(**kwargs))
        except Exception:
            entry = compose_error_entry(
                message=tb.format_exc(), endpoint=api_function.__name__, **kwargs
            )
            self.error_log.append(entry)
            return None
        return response

    def get_series_from_tag_name(self, tag_name: str) -> dict[str, FredSeries] | None:
        """
        Retrieves a dictionary of FredSeries objects based on a given tag name.

        Args:
            tag_name (str): The tag name to filter the series by.

        Returns:
            dict[str, FredSeries] | None: A dictionary where the keys are the series IDs and the values are FredSeries objects.
            Returns None if the probe endpoint was not successful.
        """  # noqa: E501
        OFFSET = 0
        series_dict = {}
        successful_probe = self.probe_endpoint(
            self.client.tags, path="series", tag_names=tag_name
        )
        if successful_probe:
            while OFFSET < successful_probe.metadata["count"]:
                temp = self._api_query(
                    self.client.tags,
                    path="series",
                    tag_names=tag_name,
                    offset=OFFSET,
                    limit=self.max_items_per_request,
                    throttle=self.throttle,
                )
                if temp is None:
                    OFFSET += self.max_items_per_request
                    continue
                # Extract Series attributes
                for series in temp.parsed_payload:
                    series_dict[series.id] = series
                OFFSET += self.max_items_per_request
        else:
            return None

        return series_dict

    def get_all_series_from_Fred(self) -> dict[str, FredSeries] | None:

        all_tag_generators = self.get_all_tag_generators_from_fred()
        if all_tag_generators is None:
            return None

        all_series_dict = {}
        for generator in all_tag_generators:
            for tag in generator:
                series_dict = self.get_series_from_tag_name(tag.name)
                all_series_dict.update(series_dict) if series_dict else {}

        if not all_series_dict:
            return None
        return all_series_dict

    def _load_series(self) -> dict[str, FredSeries]:
        """
        Load and initialize FredSeries objects from a JSON file.

        Returns:
            A dictionary containing FredSeries objects, where the keys are the series IDs.
        """
        data = load_json_file(self.fred_series_info_path)  # dict
        for key in data:
            data[key] = FredSeries(data[key])
        return data

    def filter_by_popularity(
        min_popularity: int, series_dict: dict[str, FredSeries]
    ) -> dict[str, FredSeries]:
        """
        Filters a dictionary of FredSeries objects based on their popularity.

        Args:
            min_popularity (int): The minimum popularity threshold.
            series_dict (dict[str, FredSeries]): The dictionary of FredSeries objects.

        Returns:
            dict[str, FredSeries]: A filtered dictionary containing only the popular FredSeries objects, indexed by their series IDs.
        """  # noqa: E501
        popular_series_dict = {}
        for key, value in series_dict.items():
            if value.popularity > min_popularity:
                popular_series_dict[key] = value
        return popular_series_dict

    def get_observations(self, series: FredSeries, **kwargs) -> None:
        """
        Retrieves all observations for a given FredSeries object.

        Args:
            series (FredSeries): The FredSeries object to retrieve observations for.
        """
        try:
            series._get_all_observations_from_fred()
        except Exception:
            entry = compose_error_entry(
                message=tb.format_exc(),
                endpoint="series",
                path="observations",
                series_id=series.id,
            )
            self.error_log.append(entry)

    def get(
        self,
        get_series_info_from_Fred: bool = False,
    ) -> None:
        """Fetch FRED observations from Series above a `self.min_popularity` rating.

        Args:
            get_series_info_from_Fred (bool, optional): Obtain FRED series metadata from FRED. If false, gets FRED series metadata from local file. Defaults to False.
        """  # noqa: E501

        if get_series_info_from_Fred:
            all_series = (
                self.get_all_series_from_Fred()
            )  # come back to this for error tracing
            write_files({self.fred_series_info_path: all_series})
        else:
            all_series = self._load_series()
        popular_series = self.filter_by_popularity(self.min_popularity, all_series)

        print("Getting relevant observation data...")
        for id, series in tqdm(popular_series):
            self.get_observations(series)
            if series.observations is None:
                continue
            observations = [
                observation.to_flat_dict() for observation in series.observations
            ]
            self.catalog_observations_on_disk(id, series.title, observations)

        write_error_files({self.error_log_filepath: self.error_log})

    def catalog_observations_on_disk(
        self, id: str, title: str, observations: list[dict[str, str]]
    ) -> None:
        """Write observations to the correct place on disk.

        Args:
            id (str): FRED series ID.
            observations (list[dict[str, str]]): Observations from the FRED API.
        """
        bill_signed_dates, training_example_paths = self.get_training_example_info()
        earliest_obs_date = datetime.strptime(observations[0]["date"], "%Y-%m-%d")
        latest_obs_date = datetime.strptime(observations[-1]["date"], "%Y-%m-%d")

        # Write observations to file(s)
        series_was_written = False
        if training_example_paths is not None:
            for bill_signed_date, training_example_path in zip(
                bill_signed_dates, training_example_paths
            ):  # noqa: E501
                bill_signed_date = datetime.strptime(bill_signed_date, "%Y-%m-%d")
                if (earliest_obs_date < bill_signed_date < latest_obs_date) and (
                    all(
                        series := self.create_feature_and_target_series(
                            bill_signed_date, observations
                        )
                    )
                ):
                    self.write_observations_to_file(
                        series[0],
                        self.get_training_data_filepath(id, training_example_path),
                    )
                    self.write_observations_to_file(
                        series[1],
                        self.get_training_data_filepath(
                            id, training_example_path, feature=False
                        ),
                    )
                    series_was_written = True

        if not series_was_written:
            self.write_observations_to_file(observations, self.get_filepath(id, title))

    def get_filepath(self, series_id: str, title: str) -> str:
        """Get the filepath of a FRED series.

        Args:
            series_id (str): FRED series ID.

        Returns:
            str: Filepath of the FRED series.
        """
        filename = process_filename(title + f" ({series_id}).csv")
        filepath = path.join(self.fred_data_path, filename)
        return filepath

    def get_training_data_filepath(
        self, series_id: str, training_example_path: str, feature: bool = True
    ) -> str:
        """Get the feature filepath or label filepath of a FRED series.

        Args:
            series_id (str): FRED series ID.
            feature (bool, optional): Whether to get the feature filepath or the label filepath.
                Defaults to True, which returns the feature filepath.

        Returns:
            str: Filepath of the FRED series.
        """
        if feature:
            suffix = "_series.csv"
        else:
            suffix = "_label.csv"
        filename = process_filename(series_id + suffix)
        filepath = path.join(training_example_path, series_id, filename)
        return filepath

    def get_training_example_info(
        self,
    ) -> tuple[list[str], list[str]] | tuple[None, None]:
        """Get dates that all downloaded bills were signed into law and the filesystem location of the bills.

        Returns:
            List of dates that bills were signed into law, and corresponding list of filesystem locations.
        """  # noqa: E501
        if not os.path.isdir(self.training_data_path):
            return None, None

        training_example_paths = []
        bill_approval_dates = []
        pattern = r"(\d+-\d+-\d+)_\d+-\d+"
        for item in os.listdir(self.training_data_path):
            path = os.path.join(self.training_data_path, item)
            matches = re.match(pattern, item)
            if os.path.isdir(path) and matches is not None:
                training_example_paths.append(path)
                bill_approval_dates.append(matches.group(1))

        if not training_example_paths:
            return None, None
        return bill_approval_dates, training_example_paths

    def write_observations_to_file(
        observations: list[dict[str, str]], filepath: str
    ) -> None:
        """Write FRED observations to csv file.

        Args:
            observations (list[dict[str, str]]): Observations from the FRED API.
            filepath (str): File to write observations to.
        """
        filepath = truncate_filepath(filepath)
        if not path.exists(path.dirname(filepath)):
            os.makedirs(path.dirname(filepath))

        json_to_csv(observations, filepath)

    def find_date_in_observations(
        self,
        target_date: datetime,
        observations: list[dict[str, str]],
        obs_time_format: str = "%Y-%m-%d",
    ) -> int:
        """Find the index in `observations` where `target_date` is.

        Args:
            target_date (datetime): Date to search for.
            observations (list[dict[str, str]]): Observations from the FRED API.
            obs_time_format (str, optional): Time format of `observations`. Defaults to "%Y-%m-%d".

        Returns:
            Index in `observations` where `target_date` is. If `target_date` not present, return the index of the next greatest date.
        """  # noqa: E501
        start_ind, end_ind = 0, len(observations)
        while start_ind < end_ind - 1:
            i = int((start_ind + end_ind) / 2)
            observation_date = datetime.strptime(
                observations[i]["date"], obs_time_format
            )
            if target_date == observation_date:
                return i
            elif target_date < observation_date:
                end_ind = i
                greater_flag = False
            elif target_date > observation_date:
                start_ind = i
                greater_flag = True

            if not (start_ind < end_ind - 1):
                if greater_flag:
                    i = i + 1
        return i

    def create_feature_and_target_series(
        self,
        separation_date: datetime,
        observations: list[dict[str, str]],
        feature_series_timespan: timedelta = timedelta(days=365 * 10),
        target_series_timespan: timedelta = timedelta(days=365 * 5),
    ) -> tuple[list[dict[str, str]], list[dict[str, str]]] | tuple[None, None]:
        """Split FRED observations into two datasets (`feature_series` and `target_series`) at `separation_date`. `feature_series` is then obtained from observations before `separation_date`, and `target_series` from observations after `separation_date`.

        Args:
            separation_date (datetime): Date at which `observations` are split into feature and target series.
            observations (list[dict[str, str]]): Observations from the FRED API.
            feature_series_timespan (timedelta, optional): Observations within this duration before `separation_date` become `feature_series`. Defaults to 10 years.
            target_series_timespan (timedelta, optional): Observations within this duration after `separation_date` become `target_series`. Defaults to 5 years.

        Returns:
            Tuple of `feature_series` and `target_series`.
        """  # noqa: E501
        obs_time_format = "%Y-%m-%d"

        # Calculate time interval between observations
        obs_interval = datetime.strptime(
            observations[1]["date"], obs_time_format
        ) - datetime.strptime(observations[0]["date"], obs_time_format)
        backward_steps = feature_series_timespan // obs_interval
        forward_steps = target_series_timespan // obs_interval

        # Find index of separation_date with BST
        i = self.find_date_in_observations(
            separation_date, observations, obs_time_format
        )

        # Create feature and target series
        if backward_steps <= i <= len(observations) - forward_steps:
            feature_series = observations[i - backward_steps : i]
            target_series = observations[i : i + forward_steps]
            return feature_series, target_series
        return None, None
