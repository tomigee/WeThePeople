# This file contains a script that compiles:
# 1. All passed legislature in the United States since 1947

# CONGRESS DATA
# 1. Scrape all bill data and download it locally. Amazon S3 doesn't give me much free storage
# sadly.
# - One bill per text file
#
# -------------------------------------------------------------------------------------------

import traceback as tb
import re
from json import loads, dump
from datetime import datetime

from .Congress.congress.core import Congress
from .utils import compose_error_entry
from .apidata import get_law_text


def get_public_law_number(phrase):
    pattern = r"Became Public Law No: [\d]+-([\d]+)."
    re_match = re.match(pattern, phrase)
    if re_match is not None:
        pl_num = re_match.group(1)
        return pl_num
    else:
        return None


def download_public_law(bill, client, billtext_error_log):
    latest_action = bill['latestAction']['text']
    if "became public law" in latest_action.lower():
        try:
            congress_number = bill['congress']
            bill_type = bill['type']
            bill_number = bill['number']
            law_number = get_public_law_number(latest_action)
            bill_title = f"{congress_number}-{law_number}"
            data_folder_name = bill['latestAction']['actionDate'] + "_" + bill_title
            get_law_text(
                congress_number,
                law_number,
                bill_title,
                bill_type,
                bill_number,
                client,
                f"Public Laws/{data_folder_name}"
            )
        except Exception:
            # Log error to a json file
            error_entry = compose_error_entry(
                message=tb.format_exc(),
                bill=bill
            )
            billtext_error_log.append(error_entry)


def get_list_of_bills(client, OFFSET, LIMIT, requests_error_log):
    errored_out = False
    try:
        response = loads(
            client.bill(
                offset=OFFSET,
                limit=LIMIT,
                throttle=True
            )
        )
    except Exception:
        error_entry = compose_error_entry(
            message=tb.format_exc(),
            offset=OFFSET,
            limit=LIMIT
        )
        requests_error_log.append(error_entry)
        errored_out = True
        response = None

    return response, errored_out


def download_public_law_after_date(earliest_date, bill, client, billtext_error_log):
    passed_earliest_date = False
    latest_action_date = datetime.strptime(
        bill['latestAction']['actionDate'],
        "%Y-%m-%d"
    )
    if latest_action_date > earliest_date:
        download_public_law(bill, client, billtext_error_log)
    else:
        passed_earliest_date = True

    return passed_earliest_date, latest_action_date


def script_to_run():
    client = Congress()
    OFFSET = 0
    LIMIT = 250
    search_limit = datetime(1947, 1, 1, 00, 00, 00)
    latest_action_date = datetime.now()
    finished_searching = False
    while_loop_count = 0
    requests_error_log = []
    billtext_error_log = []

    while not finished_searching:
        while_loop_count += 1

        # Query the Congress API
        list_of_bills, errored_out = get_list_of_bills(client, OFFSET, LIMIT, requests_error_log)
        if errored_out:
            OFFSET += LIMIT
            if while_loop_count % 5 == 0:
                print("Current latest action date: ", latest_action_date)
            continue

        for bill in list_of_bills['bills']:
            finished_searching, latest_action_date = download_public_law_after_date(
                search_limit,
                bill,
                client,
                billtext_error_log
            )

        OFFSET += LIMIT
        if while_loop_count % 5 == 0:
            print("Current latest action date: ", latest_action_date)

    if requests_error_log:
        with open("error_requests.json", 'w') as file:
            dump(requests_error_log, file)
    if billtext_error_log:
        with open("error_bill_text.json", 'w') as file:
            dump(billtext_error_log, file)
