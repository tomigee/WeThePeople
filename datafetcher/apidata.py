import requests
from json import loads, dumps

from .Congress.congress.core import Congress
from .utils import strip_tags


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


def get_bill_text(
    congress_num,
    bill_type,
    bill_num,
    indexing_key="type",
    indexing_value="Enrolled Bill",
    text_version_type="Formatted Text"
):

    args = [str(congress_num),
            str(bill_type),
            str(bill_num),
            "text"]
    path = "/".join(args)
    response = loads(Congress().bill(path=path))

    for bill_version in response["textVersions"]:
        if bill_version[indexing_key] == indexing_value:
            text_versions = bill_version["formats"]
            for version in text_versions:
                if version["type"] == text_version_type:
                    bill_url = version["url"]
                    break
            break

    temp = requests.get(bill_url)
    bill_text = strip_tags(temp.text)
    return bill_text
