import os
from os import path

import requests
from json import loads, dumps

from .utils import strip_tags, process_filename, truncate_filepath


class ApiData:
    def __init__(self, source, data, size=None):
        if size is None:
            if isinstance(data, str):
                self._size = len(data.encode('utf-8'))
            elif isinstance(data, dict):
                self._size = len(dumps(data).encode('utf-8'))
            else:
                raise TypeError("Unknown data type")  # delete later
        else:
            self._size = size
        self._source = source.lower()
        self._data = data
        self.__parse_data()

    def __repr__(self):
        return f"ApiData object of size {self.size} bytes"

    @property
    def size(self):
        return self._size

    @property
    def source(self):
        return self._source

    @property
    def data(self):
        return self._data

    @property
    def parsed_data(self):
        return self._parsed_data

    def __parse_data(self):
        if self._source == "bea":
            self._parsed_data = loads(self.data)["BEAAPI"]["Results"]["Data"]
            return self._parsed_data
        elif self._source == "fred":
            self._parsed_data = self.data["observations"]
            return self._parsed_data
        elif self._source == "congress":
            pass
        else:
            raise NotImplementedError(
                f"Unable to parse data; data source {self._source} has not been implemented."
            )


def get_law_text(
    congress_num,
    law_num,
    law_title,
    bill_type,
    bill_number,
    write_to_folder=None
):

    def compose_law_url(congress_num, law_num):
        url = f"https://www.congress.gov/{congress_num}/plaws/publ{law_num}/PLAW-{congress_num}publ{law_num}.htm"  # noqa E501
        return url

    def compose_bill_url(congress_num, bill_type, bill_number):
        url = f"https://www.congress.gov/{congress_num}/bills/{bill_type.lower()}{bill_number}/BILLS-{congress_num}{bill_type.lower()}{bill_number}enr.htm"  # noqa E501
        return url

    def get_clean_text(url):
        response = requests.get(url)
        clean_text = strip_tags(response.text)
        return clean_text

    law_title = process_filename(law_title)

    if law_num is not None:
        law_text = get_clean_text(
            compose_law_url(congress_num, law_num)
        )

        # Check if law_text is legit. TODO: Find a better way to do this.
        test_string = ".main-wrapper {overflow: visible !important;}"
        if test_string in law_text:
            # law_text isn't legit. Try another url
            law_text = get_clean_text(
                compose_bill_url(congress_num, bill_type, bill_number)
            )

    else:
        law_text = get_clean_text(
            compose_bill_url(congress_num, bill_type, bill_number)
        )

    # Write to file or return
    if write_to_folder is None:
        return law_text
    else:
        file_path = "/".join([str(write_to_folder), law_title+".txt"])
        if len(file_path) > 255:
            file_path = truncate_filepath(file_path)
        folder_path = path.join(os.getcwd(), write_to_folder)
        if not path.exists(folder_path):
            os.mkdir(folder_path)

        with open(file_path, 'w') as file:
            file.write(law_text)
