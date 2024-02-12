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

from .fred.fred.core import Fred
from .utils import json_to_csv, truncate_filepath, process_filename, write_to_error_file


def script_to_run():
    # Parameters for script
    LIMIT = 1000
    POPULARITY = 85
    OFFSET = 0
    get_series_info_from_Fred = False

    client = Fred()

    if get_series_info_from_Fred:
        # Get all tag names
        print("Getting all tag names...")
        all_tags = []
        temp = client.tags(limit=2)
        count = temp["count"]
        while OFFSET < count:
            temp = client.tags(limit=LIMIT, offset=OFFSET, throttle=True)
            for tag in temp['tags']:
                all_tags.append(tag['name'])
            OFFSET += LIMIT

        # Get all series
        print("Getting all series info...")
        series_dict = {}
        for tag in all_tags:
            OFFSET = 0
            try:
                temp1 = client.tags('series', tag_names=tag, limit=2, throttle=True)
            except Exception:
                # Log attributes to a text file
                write_to_error_file(f"Exception encountered:{tb.format_exc()}", False)
                if 'tag' in locals():
                    write_to_error_file(f"tag: {tag}")
                if 'temp1' in locals():
                    write_to_error_file(f"temp1: {temp1}")
                continue

            count = temp1['count']

            while OFFSET < count:
                try:
                    temp = client.tags(
                        'series', tag_names=tag, offset=OFFSET,
                        limit=LIMIT, throttle=True
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
                    # Log attributes to a text file
                    write_to_error_file(f"Exception encountered:{tb.format_exc()}", False)
                    if 'tag' in locals():
                        write_to_error_file(f"tag: {tag}")
                    if 'count' in locals():
                        write_to_error_file(f"count: {count}")
                    if 'OFFSET' in locals():
                        write_to_error_file(f"OFFSET: {OFFSET}")
                    if 'temp' in locals():
                        write_to_error_file(f"temp: {temp}")
                    break

                OFFSET += LIMIT

        with open("fred_series_info.json", "w") as file:
            dump(series_dict, file)
    else:
        with open("fred_series_info.json", "r") as file:
            series_dict = load(file)

    print("length of series_dict:", len(series_dict))
    # Get top series by popularity
    popular_series_ids = []
    for id, details in series_dict.items():
        if details['popularity'] >= POPULARITY:
            popular_series_ids.append(id)

    print("Getting relevant observation data...")
    for id in popular_series_ids:
        try:
            temp1 = client.series('observations', series_id=id, limit=2, throttle=True)
        except Exception:
            # Log attributes to a text file
            write_to_error_file(f"Exception encountered:{tb.format_exc()}", False)
            if 'id' in locals():
                write_to_error_file(f"id: {id}")
            if 'temp1' in locals():
                write_to_error_file(f"temp1: {temp1}")
            continue

        count = temp1['count']
        observations = []
        OFFSET = 0

        while OFFSET < count:
            try:
                temp = client.series(
                    'observations', series_id=id, offset=OFFSET, limit=LIMIT, throttle=True
                )
                observations += temp['observations']
            except Exception:
                # Log attributes to a text file
                write_to_error_file(f"Exception encountered:{tb.format_exc()}", False)
                if 'id' in locals():
                    write_to_error_file(f"id: {id}")
                if 'OFFSET' in locals():
                    write_to_error_file(f"OFFSET: {OFFSET}")
                if 'temp' in locals():
                    write_to_error_file(f"temp: {temp}")
                break

            OFFSET += LIMIT

        filename = process_filename(
            series_dict[id]['title'] + f" ({id})"
        )
        filepath = path.join(
            "FredData",
            filename
        )
        filepath = truncate_filepath(filepath)
        if not path.exists(path.dirname(filepath)):
            os.mkdir(path.dirname(filepath))

        json_to_csv(observations, filepath)
