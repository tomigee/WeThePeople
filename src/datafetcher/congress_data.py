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
    load_json_file,
)


class CongressData:
    """A class for fetching and processing data from the Congress.gov API.

    Args:
        search_limit (datetime, optional): The limit for the search protocol, specifying the earliest date for bills to be downloaded. Defaults to datetime(1947, 1, 1, 00, 00, 00).
        max_consec_error_count (int, optional): The maximum number of consecutive errors allowed before stopping the search protocol. Defaults to 200.
        limit (int, optional): The limit for the number of bills to be fetched per API request. Defaults to 250.
        start_from_checkpoint (bool, optional): Indicates whether to start the search protocol from the last checkpoint. Defaults to True.
        training_data_path (str, optional): The path to the folder where training data will be stored. Defaults to "datasets/training data".
        requests_error_path (str, optional): The path to the file where requests error logs will be stored. Defaults to "error logs/error_requests.json".
        billtext_error_path (str, optional): The path to the file where bill text error logs will be stored. Defaults to "error logs/error_bill_text.json".
        checkpoint_file_path (str, optional): The path to the file where the checkpoint data will be stored. Defaults to "error logs/bill_requests_checkpoint.json".
    """  # noqa: E501

    def __init__(
        self,
        search_limit: datetime = datetime(1947, 1, 1, 00, 00, 00),
        max_consec_error_count: int = 200,
        limit: int = 250,
        start_from_checkpoint: bool = True,
        training_data_path: str = "datasets/training data",
        requests_error_path: str = "error logs/error_requests.json",
        billtext_error_path: str = "error logs/error_bill_text.json",
        checkpoint_file_path: str = "error logs/bill_requests_checkpoint.json",
    ):
        self.search_limit = search_limit
        self.max_consec_error_count = max_consec_error_count
        if limit > 250:
            self.limit = 250
        else:
            self.limit = limit

        self.start_from_checkpoint = start_from_checkpoint
        self.client = Congress()
        self._bill = None
        self._data_folder_name = None
        self.training_data_path = training_data_path
        self.requests_error_path = requests_error_path
        self.billtext_error_path = billtext_error_path
        self.checkpoint_file_path = checkpoint_file_path

    @property
    def data_folder_name(self):
        return self._data_folder_name

    @property
    def bill(self):
        return self._bill

    @bill.setter
    def bill(self, new_bill):
        terminal_folder = (
            new_bill.raw["latestAction"]["actionDate"] + "_" + new_bill.bill_title
        )
        self._data_folder_name = f"{self.training_data_path}/{terminal_folder}"
        self._bill = new_bill

    def _get_clean_law_text(self, url: str, max_retries: int = 3) -> str:
        """Gets clean law/bill text from provided URL. Strips HTML tags.

        Args:
            url (str): URL where raw text is located.
            max_retries (int, optional): Maximum number of retries if the law text is bad. Defaults to 3.

        Returns:
            str: Cleaned law/bill text.
        """  # noqa: E501
        clean_text = None
        retries = 0
        law_text_is_bad = self._is_law_text_bad(clean_text)
        while law_text_is_bad and retries < max_retries:
            response = requests.get(url)
            clean_text = strip_tags(response.text)
            law_text_is_bad = self._is_law_text_bad(clean_text)
            retries += 1
        if law_text_is_bad:
            return None
        return clean_text

    def _is_law_text_bad(self, law_text: str) -> bool:
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

    def _write_to_disk(self, law_text: str) -> None:
        """Writes the law text to disk.

        Args:
            write_to_folder (str): Folder to which raw text files will be written.
        """
        law_title = process_filename(self.bill_title)
        file_path = "/".join([str(self.data_folder_name), law_title + ".txt"])
        if len(file_path) > 255:
            file_path = truncate_filepath(file_path)
        folder_path = path.join(os.getcwd(), self.data_folder_name)
        if not path.exists(folder_path):
            os.makedirs(folder_path)

        with open(file_path, "w") as file:
            file.write(law_text)

    def _get_law_text(self) -> str:
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
        """  # noqa: E501

        if self.bill.law_number is not None:
            law_text = self._get_clean_law_text(self.bill.law_url)
        else:
            law_text = self._get_clean_law_text(self.bill.bill_url)

        if law_text is None:
            raise ValueError("Bad API Response")
        return law_text

    def download_public_law(self) -> None:
        """Downloads the public law version of a given bill (if possible).

        Args:
            bill (dict[str, ]): Information about the given bill, as provided by Congress.gov API
            client (Congress): The Congress client object used to send requests to the Congress.gov API.
            billtext_error_log (list[dict[str,]]): Log of detailed errors encountered when downloading the bill text.
        """  # noqa: E501

        if "became public law" in self.bill.latest_action.lower():
            try:
                law_text = self._get_law_text()
                self._write_to_disk(law_text)
            except Exception:
                # Log error to a json file
                error_entry = compose_error_entry(
                    message=tb.format_exc(), bill=self.bill.raw
                )
                self.billtext_error_log.append(error_entry)

    def get_list_of_bills(self) -> tuple[dict[str, list[dict[str,]]], bool]:
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
            response = loads(
                self.client.bill(offset=self.OFFSET, limit=self.limit, throttle=True)
            )
        except Exception:
            error_entry = compose_error_entry(
                message=tb.format_exc(), offset=self.OFFSET, limit=self.limit
            )
            self.requests_error_log.append(error_entry)
            errored_out = True
            response = None

        return response, errored_out

    def download_public_law_if_after_date(self) -> tuple[bool, datetime]:
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
        latest_action_date = datetime.strptime(self.bill.latest_action_date, "%Y-%m-%d")
        if latest_action_date > self.search_limit:
            self.download_public_law()
        else:
            passed_earliest_date = True

        return passed_earliest_date, latest_action_date

    def _initialize_variables(self):
        self.latest_action_date = datetime.now()
        self.finished_searching = False
        self.while_loop_count = 0
        self.consec_error_count = 0
        self.OFFSET = (
            0
            if not self.start_from_checkpoint
            else self._load_checkpoint()["offset"]
            - self.max_consec_error_count * self.limit
        )
        self.requests_error_log = (
            []
            if not self.start_from_checkpoint
            else self._load_error_log(self.requests_error_path)
        )
        self.billtext_error_log = (
            []
            if not self.start_from_checkpoint
            else self._load_error_log(self.billtext_error_path)
        )

    def _load_error_log(self, path):
        with open(path, "r") as file:
            return load(file)

    def _load_checkpoint(self):
        return load_json_file(self.checkpoint_file_path)

    def _handle_error(self):
        self.consec_error_count += 1
        if self.consec_error_count > self.max_consec_error_count:
            temp = {
                "offset": self.OFFSET,
                "stoptime": datetime.strftime(datetime.now(), "%Y-%m-%d %H:%M:%S"),
            }
            with open(self.checkpoint_file_path, "w") as file:
                dump(temp, file)  # save checkpoint file
            return True
        self.OFFSET += self.limit
        return False

    def _process_bills(self, list_of_bills):
        if not list_of_bills["bills"]:
            self.finished_searching = True
        for bill in list_of_bills["bills"]:
            self.bill = CongressBill(bill)
            self.finished_searching, self.latest_action_date = (
                self.download_public_law_if_after_date(self.billtext_error_log)
            )
            if self.finished_searching:
                break
        self.OFFSET += self.limit

    def get(self) -> None:
        """Get law text for all bills that became public law after `search_limit`.

        Args:
            max_items_per_request (int, optional): Congress.gov API parameter; maximum number of records that each request can return. Defaults to 250.
            start_from_checkpoint (bool, optional): Indicate if starting from a checkpoint. If False, starts searching from today's date. Defaults to True.
            search_limit (datetime, optional): Date after which bills signed into law should be downloaded. Defaults to datetime(1947, 1, 1, 00, 00, 00).
            max_consec_error_count (int, optional): Maximum number of times this function can error out due to API issues. Defaults to 200.
        """  # noqa: E501
        self._initialize_variables()

        while not self.finished_searching:
            self.while_loop_count += 1
            list_of_bills, errored_out = self.get_list_of_bills()
            if errored_out:
                if self._handle_error():
                    break
                continue
            self.consec_error_count = 0
            self._process_bills(list_of_bills)
            if self.while_loop_count % 5 == 0:
                print("Current latest action date: ", self.latest_action_date)

        write_error_files(
            {
                self.requests_error_path: self.requests_error_log,
                self.billtext_error_path: self.billtext_error_log,
            }
        )

    def _print_error_logs_summary(
        self,
        requests_error_log: list[dict[str,]],
        billtext_error_log: list[dict[str,]],
    ) -> None:
        """Prints a summary of the error logs."""
        print("Requests error log length before retrying:", len(requests_error_log))
        print(
            "Requests error log length after retrying:",
            len(self.requests_error_log),
        )
        print("Billtext error log length before retrying:", len(billtext_error_log))
        print(
            "Billtext error log length after retrying:",
            len(self.billtext_error_log),
        )

    def retry_errors(self) -> None:
        """Retry failed attempts at fetching bill info and/or downloading bill text."""  # noqa: E501
        self.requests_error_log = []
        self.billtext_error_log = []
        requests_error_log = self._load_error_log(self.requests_error_path)
        billtext_error_log = self._load_error_log(self.billtext_error_path)

        for request in tqdm(requests_error_log, desc="Retrying requests"):
            self.OFFSET = request["offset"]
            self.limit = request["limit"]
            list_of_bills, errored_out = self.get_list_of_bills()
            if errored_out:
                continue
            self._process_bills(list_of_bills)

        for item in tqdm(billtext_error_log, desc="Retrying billtext"):
            self.bill = CongressBill(item["bill"])
            self.download_public_law()

        self._print_error_logs_summary(requests_error_log, billtext_error_log)

        write_error_files(
            {
                self.requests_error_path: self.requests_error_log,
                self.billtext_error_path: self.billtext_error_log,
            }
        )


# -------------------------------------------------------------------------------------------


class CongressBill:
    def __init__(self, bill: dict[str,]) -> None:
        self.raw = bill
        self.congress_number = bill["congress"]
        self.bill_type = bill["type"]
        self.bill_number = bill["number"]
        self.law_number = self.get_public_law_number(bill["latestAction"]["text"])
        self.bill_title = f"{self.congress_number}-{self.bill_number}"
        self.latest_action = bill["latestAction"]["text"]
        self.latest_action_date = bill["latestAction"]["actionDate"]
        self._law_url = None
        self._bill_url = None
        self.client = Congress()

    def _get_public_law_number(self, phrase: str) -> str | None:
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

    def _get_text_versions(self):
        res = loads(
            self.client.bill(
                f"{self.raw.congress_number}/{self.raw.bill_type}/{self.raw.bill_number}/text",
                throttle=True,
            )
        )
        return res["textVersions"]

    @property
    def law_url(self) -> str:
        """Composes URL where law text is located.

        Args:
            congress_num (str): The congress number. For example, for the 117th congress, this
            argument is 117.
            law_num (str): The public law number. Appended to the name of the downloaded file.

        Returns:
            URL where law text is located.
        """
        if not self._law_url:
            self._law_url = f"https://www.congress.gov/{self.congress_number}/plaws/publ{self.law_number}/PLAW-{self.congress_number}publ{self.law_number}.htm"  # noqa E501
        return self._law_url

    @property
    def bill_url(self) -> str:
        """Composes URL where bill text is located.

        Args:
            congress_num (str): _description_
            bill_type (str): _description_
            bill_number (str): _description_

        Returns:
            str: _description_
        """
        if not self._bill_url:
            self._bill_url = f"https://www.congress.gov/{self.congress_number}/bills/{self.bill_type.lower()}{self.bill_number}/BILLS-{self.congress_number}{self.bill_type.lower()}{self.bill_number}enr.htm"  # noqa E501
        return self._bill_url
