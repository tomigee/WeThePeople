# This file contains a script that compiles:
# 1. All passed legislature in the United States since 1947
# -------------------------------------------------------------------------------------------

import os
from os import path
import traceback as tb
import re
from json import loads, dump, load
from datetime import datetime

import requests
from tqdm import tqdm

from .Congress.congress.core import Congress
from .utils import (
    compose_error_entry,
    write_error_files,
    strip_tags,
    process_filename,
    truncate_filepath,
)

# Script parameters
training_data_path = "datasets/training data"
requests_error_path = "error logs/error_requests.json"
billtext_error_path = "error logs/error_bill_text.json"
checkpoint_file_path = "error logs/bill_requests_checkpoint.json"


def get_law_text(
    congress_num: str,
    law_num: str,
    law_title: str,
    bill_type: str,
    bill_number: str,
    client: Congress,
    write_to_folder: str = None,
) -> str | None:
    """Downloads or returns the raw text of a public law. Currently supports Formatted Text
    Congress.gov format only. Support for PDF might be added later.

    Args:
        congress_num (str): The congress number. For example, for the 117th congress, this
        argument is 117.
        law_num (str): The public law number. Appended to the name of the downloaded file.
        law_title (str): The title of the law. Appended to the name of the downloaded file.
        bill_type (str): The bill type. See https://api.congress.gov/#/bill/bill_list_by_type
        for details.
        bill_number (str): The bill number. See https://api.congress.gov/#/bill/bill_details
        for details.
        client (Congress): The Congress client object used to send requests to the Congress.gov API.
        write_to_folder (str, optional): Folder to which raw text files will be written. If not
        provided, raw text will be returned. Defaults to None.

    Returns:
        The law text if `write_to_folder` is not provided. `None` otherwise.
    """

    def compose_law_url(congress_num: str, law_num: str) -> str:
        """Composes URL where law text is located.

        Args:
            congress_num (str): The congress number. For example, for the 117th congress, this
            argument is 117.
            law_num (str): The public law number. Appended to the name of the downloaded file.

        Returns:
            URL where law text is located.
        """
        url = f"https://www.congress.gov/{congress_num}/plaws/publ{law_num}/PLAW-{congress_num}publ{law_num}.htm"  # noqa E501
        return url

    def compose_bill_url(congress_num: str, bill_type: str, bill_number: str) -> str:
        """Composes URL where bill text is located.

        Args:
            congress_num (str): _description_
            bill_type (str): _description_
            bill_number (str): _description_

        Returns:
            str: _description_
        """
        url = f"https://www.congress.gov/{congress_num}/bills/{bill_type.lower()}{bill_number}/BILLS-{congress_num}{bill_type.lower()}{bill_number}enr.htm"  # noqa E501
        return url

    def get_clean_text(url: str) -> str:
        """Removes HTML tags from raw text.

        Args:
            url (str): URL where raw text is located.

        Returns:
            Text with HTML tags removed
        """
        response = requests.get(url)
        clean_text = strip_tags(response.text)
        return clean_text

    def is_law_text_bad(law_text: str) -> bool:
        """Detects if the downloaded law text is garbage error text.

        Args:
            law_text (str): The raw law text.

        Returns:
            True if the law text is unintelligible. False otherwise.
        """
        # TODO: Find a better way to do this.
        if law_text is None:
            return True

        test_string = "Congress.gov | Library of Congress"
        test_string1 = "Just a moment"
        test_string2 = "www.congress.gov | 522: Connection timed out"
        law_text_is_bad = (
            (test_string in law_text[:42])
            or (test_string1 in law_text[:13])
            or (test_string2 in law_text[:54])
        )
        return law_text_is_bad

    def get_bill_file_format(formats: list[dict[str, str]]) -> str | None:
        """Get the format of the bill.

        Args:
            formats (list[dict[str, str]]): TODO: Populate this.

        Returns:
            The format of the bill.
        """
        for file_format in formats:
            if file_format["type"] == "Formatted Text":
                return "Formatted Text"
            elif file_format["type"] == "PDF":
                ans = "PDF"
            else:
                ans = None
        return ans

    law_title = process_filename(law_title)

    if law_num is not None:
        law_text = get_clean_text(compose_law_url(congress_num, law_num))
    else:
        law_text = get_clean_text(
            compose_bill_url(congress_num, bill_type, bill_number)
        )

    # Check if law_text is legit.
    law_text_is_bad = is_law_text_bad(law_text)

    if law_text_is_bad:  # Get bill file format
        res = loads(
            client.bill(f"{congress_num}/{bill_type}/{bill_number}/text", throttle=True)
        )
        bill_file_format = None
        for text_version in res["textVersions"]:
            if text_version["type"] == "Enrolled Bill":
                bill_file_format = get_bill_file_format(text_version["formats"])
                break

    retries = 0
    while law_text_is_bad and retries < 3:  # law_text isn't legit; try bill text
        if bill_file_format == "Formatted Text":  # if formatted text exists
            law_text = get_clean_text(
                compose_bill_url(congress_num, bill_type, bill_number)
            )

        # elif bill_file_format == "PDF":
        # do something, yet to be implemented

        else:
            break

        law_text_is_bad = is_law_text_bad(law_text)
        retries += 1

    if law_text_is_bad:
        # error out
        raise ValueError("Bad API Response")

    # Write to file or return
    if write_to_folder is None:
        return law_text
    else:
        file_path = "/".join([str(write_to_folder), law_title + ".txt"])
        if len(file_path) > 255:
            file_path = truncate_filepath(file_path)
        folder_path = path.join(os.getcwd(), write_to_folder)
        if not path.exists(folder_path):
            os.makedirs(folder_path)

        with open(file_path, "w") as file:
            file.write(law_text)


def get_public_law_number(phrase: str) -> str | None:
    """Get the law number of a public law.

    Args:
        phrase (str): Phrase to check for "Became public Law" regex pattern.

    Returns:
        The public law number.
    """
    pattern = r"Became Public Law No: [\d]+-([\d]+)."
    re_match = re.match(pattern, phrase)
    if re_match is not None:
        pl_num = re_match.group(1)
        return pl_num
    else:
        return None


def download_public_law(
    bill: dict[str,], client: Congress, billtext_error_log: list[dict[str,]]
) -> None:
    """Downloads the public law version of a given bill (if possible).

    Args:
        bill (dict[str, ]): Information about the given bill, as provided by Congress.gov API
        client (Congress): The Congress client object used to send requests to the Congress.gov API.
        billtext_error_log (list[dict[str,]]): Log of detailed errors encountered when downloading the bill text.
    """  # noqa: E501
    latest_action = bill["latestAction"]["text"]
    if "became public law" in latest_action.lower():
        try:
            congress_number = bill["congress"]
            bill_type = bill["type"]
            bill_number = bill["number"]
            law_number = get_public_law_number(latest_action)
            bill_title = f"{congress_number}-{law_number}"
            data_folder_name = bill["latestAction"]["actionDate"] + "_" + bill_title
            get_law_text(
                congress_number,
                law_number,
                bill_title,
                bill_type,
                bill_number,
                client,
                f"{training_data_path}/{data_folder_name}",
            )
        except Exception:
            # Log error to a json file
            error_entry = compose_error_entry(message=tb.format_exc(), bill=bill)
            billtext_error_log.append(error_entry)


def get_list_of_bills(
    client: Congress, OFFSET: int, LIMIT: int, requests_error_log: list[dict[str, str]]
) -> tuple[dict[str, list[dict[str,]]], bool]:
    """Get metadata on bills by querying Congress.gov API bills endpoint with the given `OFFSET`
    and `LIMIT` parameters.

    Args:
        client (Congress): The Congress client object used to send requests to the Congress.gov API.
        OFFSET (int): Congress.gov API parameter. See https://api.congress.gov/#/bill/bill_list_all for details.
        LIMIT (int): Congress.gov API parameter. See https://api.congress.gov/#/bill/bill_list_all for details.
        requests_error_log (list[dict[str, str]]): Log of errors encountered when querying the Congress.gov API bills endpoint.

    Returns:
        A tuple containing the de-serialized JSON response from the API and an indicator of whether the transaction was successful.
    """  # noqa: E501
    errored_out = False
    try:
        response = loads(client.bill(offset=OFFSET, limit=LIMIT, throttle=True))
    except Exception:
        error_entry = compose_error_entry(
            message=tb.format_exc(), offset=OFFSET, limit=LIMIT
        )
        requests_error_log.append(error_entry)
        errored_out = True
        response = None

    return response, errored_out


def download_public_law_after_date(
    earliest_date: datetime,
    bill: dict[str,],
    client: Congress,
    billtext_error_log: list[dict[str,]],
) -> tuple[bool, datetime]:
    """Downloads all available public laws signed into law after `earliest_date`.

    Args:
        earliest_date (datetime): The date after which bills signed into law are to be downloaded.
        bill (dict[str, ]): Information about the given bill, as provided by Congress.gov API
        client (Congress): The Congress client object used to send requests to the Congress.gov API.
        billtext_error_log (list[dict[str,]]): Log of detailed errors encountered when downloading the bill text.

    Returns:
        Indicator of whether the search protocol has arrived at `earliest_date`, the latest date the search protocol has examined.
    """  # noqa: E501
    passed_earliest_date = False
    latest_action_date = datetime.strptime(
        bill["latestAction"]["actionDate"], "%Y-%m-%d"
    )
    if latest_action_date > earliest_date:
        download_public_law(bill, client, billtext_error_log)
    else:
        passed_earliest_date = True

    return passed_earliest_date, latest_action_date


def get(
    max_items_per_request: int = 250,
    start_from_checkpoint: bool = True,
    search_limit: datetime = datetime(1947, 1, 1, 00, 00, 00),
    max_consec_error_count: int = 200,
) -> None:
    """Get law text for all bills that became public law after `search_limit`.

    Args:
        max_items_per_request (int, optional): Congress.gov API parameter; maximum number of records that each request can return. Defaults to 250.
        start_from_checkpoint (bool, optional): Indicate if starting from a checkpoint. If False, starts searching from today's date. Defaults to True.
        search_limit (datetime, optional): Date after which bills signed into law should be downloaded. Defaults to datetime(1947, 1, 1, 00, 00, 00).
        max_consec_error_count (int, optional): Maximum number of times this function can error out due to API issues. Defaults to 200.
    """  # noqa: E501
    # Initialize variables
    if max_items_per_request > 250:
        LIMIT = 250
    else:
        LIMIT = max_items_per_request
    client = Congress()
    latest_action_date = datetime.now()
    finished_searching = False
    while_loop_count = 0
    consec_error_count = 0
    if start_from_checkpoint:
        with open(requests_error_path, "r") as file:
            requests_error_log = load(file)
        with open(billtext_error_path, "r") as file:
            billtext_error_log = load(file)
        with open(checkpoint_file_path, "r") as file:
            temp = load(file)
        OFFSET = temp["offset"] - max_consec_error_count * LIMIT
    else:
        OFFSET = 0
        requests_error_log = []
        billtext_error_log = []

    while not finished_searching:
        while_loop_count += 1

        # Query the Congress API
        list_of_bills, errored_out = get_list_of_bills(
            client, OFFSET, LIMIT, requests_error_log
        )
        if errored_out:
            consec_error_count += 1
            if (
                consec_error_count > max_consec_error_count
            ):  # HARD STOP if it keeps erroring out
                # TODO: Write offset to file the first time it errors out with code 429
                temp = {
                    "offset": OFFSET,
                    "stoptime": datetime.strftime(datetime.now(), "%Y-%m-%d %H:%M:%S"),
                }
                with open(checkpoint_file_path, "w") as file:
                    dump(temp, file)  # save checkpoint file
                break

            OFFSET += LIMIT
            if while_loop_count % 5 == 0:
                print("Current latest action date: ", latest_action_date)
            continue
        consec_error_count = 0

        if not list_of_bills["bills"]:
            finished_searching = True

        for bill in list_of_bills["bills"]:
            finished_searching, latest_action_date = download_public_law_after_date(
                search_limit, bill, client, billtext_error_log
            )

        OFFSET += LIMIT
        if while_loop_count % 5 == 0:
            print("Current latest action date: ", latest_action_date)

    write_error_files(
        {
            requests_error_path: requests_error_log,
            billtext_error_path: billtext_error_log,
        }
    )


def retry_errors(search_limit: datetime = datetime(1947, 1, 1, 00, 00, 00)) -> None:
    """Retry failed attempts at fetching bill info and/or downloading bill text.

    Args:
        search_limit (datetime, optional): Date after which bills signed into law should be downloaded. Defaults to datetime(1947, 1, 1, 00, 00, 00).
    """  # noqa: E501
    requests_error_log = []

    with open(requests_error_path, "r") as file:
        error_requests = load(file)
    no_of_request_errors = len(error_requests)
    with open(billtext_error_path, "r") as file:
        billtext_error_log = load(file)

    client = Congress()
    print("Retrying request errors...")

    # Tackle requests error log
    for request in tqdm(error_requests):
        OFFSET = request["offset"]
        LIMIT = request["limit"]
        list_of_bills, errored_out = get_list_of_bills(
            client, OFFSET, LIMIT, requests_error_log
        )

        if errored_out:
            continue

        for bill in list_of_bills["bills"]:
            finished_searching, _ = download_public_law_after_date(
                search_limit, bill, client, billtext_error_log
            )
            if finished_searching:
                break

    print("Length of requests_error_log before retrying:", no_of_request_errors)
    print("Length of requests_error_log after retrying:", len(requests_error_log))

    # Tackle billtext error log
    no_of_billtext_errors = len(billtext_error_log)
    new_billtext_error_log = []

    for item in tqdm(billtext_error_log):
        download_public_law(item["bill"], client, new_billtext_error_log)

    print("Length of billtext_error_log before retrying:", no_of_billtext_errors)
    print("Length of billtext_error_log after retrying:", len(new_billtext_error_log))

    if requests_error_log:
        with open(requests_error_path, "w") as file:
            dump(requests_error_log, file)
    if new_billtext_error_log:
        with open(billtext_error_path, "w") as file:
            dump(new_billtext_error_log, file)
