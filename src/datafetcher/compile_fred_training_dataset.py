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
from json import dump, load
import traceback as tb
import re
from datetime import datetime, timedelta

from tqdm import tqdm

from .fred.fred.core import Fred
from .utils import (
    json_to_csv,
    truncate_filepath,
    process_filename,
    compose_error_entry,
    write_error_files
)

# Script parameters
training_data_path = "datasets/training data"
fred_data_path = "datasets/FredData"
fred_series_info_path = "datasets/fred_series_info.json"
tags_requests_error_filepath = "error logs/error_tags_requests.json"
tags_data_error_filepath = "error logs/error_tags_data.json"
observations_requests_error_filepath = "error logs/error_observations_requests.json"
observations_data_error_filepath = "error logs/error_observations_data.json"


def write_observations_to_file(observations, filepath):
    filepath = truncate_filepath(filepath)
    if not path.exists(path.dirname(filepath)):
        os.makedirs(path.dirname(filepath))

    json_to_csv(observations, filepath)


def find_date_in_observations(target_date, observations, obs_time_format="%Y-%m-%d"):
    start_ind, end_ind = 0, len(observations)
    while start_ind < end_ind-1:
        i = int((start_ind + end_ind)/2)
        observation_date = datetime.strptime(observations[i]["date"], obs_time_format)
        if target_date == observation_date:
            return i
        elif target_date < observation_date:
            end_ind = i
            greater_flag = False
        elif target_date > observation_date:
            start_ind = i
            greater_flag = True

        if not (start_ind < end_ind-1):
            if greater_flag:
                i = i+1
    return i


def create_feature_and_target_series(
    separation_date,
    observations,
    feature_series_timespan=timedelta(days=365*10),
    target_series_timespan=timedelta(days=365*5)
):
    obs_time_format = "%Y-%m-%d"

    # Calculate time interval between observations
    obs_interval = datetime.strptime(
        observations[1]["date"], obs_time_format
    ) - datetime.strptime(
        observations[0]["date"], obs_time_format
    )
    backward_steps = feature_series_timespan//obs_interval
    forward_steps = target_series_timespan//obs_interval

    # Find index of separation_date with BST
    i = find_date_in_observations(separation_date, observations, obs_time_format)

    # Create feature and target series
    if backward_steps <= i <= len(observations)-forward_steps:
        feature_series = observations[i-backward_steps:i]
        target_series = observations[i:i+forward_steps]
        return feature_series, target_series
    return None, None


def get_training_example_info():
    if not os.path.isdir(training_data_path):
        return None, None

    training_example_paths = []
    bill_approval_dates = []
    pattern = r"(\d+-\d+-\d+)_\d+-\d+"
    for item in os.listdir(training_data_path):
        path = os.path.join(training_data_path, item)
        matches = re.match(pattern, item)
        if os.path.isdir(path) and matches is not None:
            training_example_paths.append(path)
            bill_approval_dates.append(matches.group(1))

    if not training_example_paths:
        return None, None
    return bill_approval_dates, training_example_paths


def script_to_run(n_max_results=1000,
                  min_popularity=85,
                  get_series_info_from_Fred=False):
    OFFSET = 0
    client = Fred()
    if n_max_results > 1000:
        n_max_results = 1000

    if get_series_info_from_Fred:
        # Get all tag names
        print("Getting all tag names...")
        all_tags = []
        temp = client.tags(limit=2)
        count = temp["count"]
        while OFFSET < count:
            temp = client.tags(limit=n_max_results, offset=OFFSET, throttle=True)
            for tag in temp['tags']:
                all_tags.append(tag['name'])
            OFFSET += n_max_results

        # Get all series
        print("Getting all series info...")
        series_dict = {}
        tags_request_error_log = []
        tags_data_error_log = []
        for tag in all_tags:
            OFFSET = 0
            try:
                temp1 = client.tags('series', tag_names=tag, limit=2, throttle=True)
            except Exception:
                entry = compose_error_entry(
                    path='series',
                    tag_names=tag,
                    limit=2,
                    throttle=True
                )
                tags_request_error_log.append(entry)
                continue

            count = temp1['count']

            while OFFSET < count:
                try:
                    temp = client.tags(
                        'series', tag_names=tag, offset=OFFSET,
                        limit=n_max_results, throttle=True
                    )
                # Extract Series ID, Title, and popularity
                    for series in temp['seriess']:
                        if series['id'] not in series_dict:
                            entry = {
                                series['id']: {
                                    'title': series['title'],
                                    'popularity': series['popularity']
                                }
                            }
                            series_dict.update(entry)
                except Exception:
                    entry = compose_error_entry(
                        path='series',
                        tag_names=tag,
                        offset=OFFSET,
                        limit=n_max_results,
                        throttle=True
                    )
                    tags_data_error_log.append(entry)
                    break

                OFFSET += n_max_results

        with open(fred_series_info_path, "w") as file:
            dump(series_dict, file)
        if tags_request_error_log:
            with open(tags_requests_error_filepath, 'w') as file:
                dump(tags_request_error_log, file)
        if tags_data_error_log:
            with open(tags_data_error_filepath, 'w') as file:
                dump(tags_data_error_log, file)
    else:
        with open("fred_series_info.json", "r") as file:
            series_dict = load(file)

    print("length of series_dict:", len(series_dict))
    # Get top series by popularity
    popular_series_ids = []
    for id, details in series_dict.items():
        if details['popularity'] >= min_popularity:
            popular_series_ids.append(id)

    print("Getting relevant observation data...")
    observations_request_error_log = []
    observations_data_error_log = []
    for id in tqdm(popular_series_ids):
        try:
            temp1 = client.series('observations', series_id=id, limit=2, throttle=True)
        except Exception:
            entry = compose_error_entry(
                message=tb.format_exc(),
                path='observations',
                series_id=id,
                limit=2,
                throttle=True
            )
            observations_request_error_log.append(entry)
            continue

        count = temp1['count']
        observations = []
        OFFSET = 0

        while OFFSET < count:
            try:
                temp = client.series(
                    'observations', series_id=id, offset=OFFSET, limit=n_max_results, throttle=True
                )
                observations += temp['observations']
            except Exception:
                entry = compose_error_entry(
                    message=tb.format_exc(),
                    path='observations',
                    series_id=id,
                    offset=OFFSET,
                    limit=n_max_results,
                    throttle=True
                )
                observations_data_error_log.append(entry)
                break

            OFFSET += n_max_results

        bill_signed_dates, training_example_paths = get_training_example_info()
        earliest_obs_date = datetime.strptime(observations[0]['date'], "%Y-%m-%d")
        latest_obs_date = datetime.strptime(observations[-1]['date'], "%Y-%m-%d")

        # Write observations to file(s)
        series_was_written = False
        if (
            training_example_paths is not None
        ):
            for bill_signed_date, training_example_path in zip(bill_signed_dates, training_example_paths):  # noqa: E501
                bill_signed_date = datetime.strptime(bill_signed_date, "%Y-%m-%d")

                if (
                    earliest_obs_date < bill_signed_date < latest_obs_date
                ) and (
                    all(series := create_feature_and_target_series(bill_signed_date, observations))
                ):
                    label_filename = process_filename(f"{id}_label.csv")
                    feature_filename = process_filename(f"{id}_series.csv")
                    feature_filepath = path.join(training_example_path, id, feature_filename)
                    label_filepath = path.join(training_example_path, id, label_filename)
                    write_observations_to_file(series[0], feature_filepath)
                    write_observations_to_file(series[1], label_filepath)
                    series_was_written = True

        if not series_was_written:
            filename = process_filename(
                series_dict[id]['title'] + f" ({id}).csv"
            )
            filepath = path.join(fred_data_path, filename)
            write_observations_to_file(observations, filepath)

    write_error_files(
        {
            observations_requests_error_filepath: observations_request_error_log,
            observations_data_error_filepath: observations_data_error_log
        }
    )
