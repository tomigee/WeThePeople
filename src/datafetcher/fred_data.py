from .fred.fred.core import Fred
from collections.abc import Generator


class FredData:
    def __init__(self, data: dict[str,]):
        if len(data) == 1:  # data is indexed_dict
            data = data[list(data)[0]]  # turn to flat_dict
        self.raw = data
        self.client = Fred()

    def to_flat_dict(self):
        return self.raw

    def _infer_index_key(self):
        return "id" if "id" in self.raw else "name"

    def to_indexed_dict(self, key: str = _infer_index_key()):
        return {self.raw[key]: self.raw}


class FredTag(FredData):
    def __init__(self, data: dict[str,]):
        super().__init__(data)
        self.name = self.raw["name"]
        self.group_id = self.raw["group_id"]
        self.notes = self.raw["notes"]
        self.created = self.raw["created"]
        self.popularity = self.raw["popularity"]
        self.series_count = self.raw["series_count"]


class FredSeries(FredData):
    def __init__(self, data: dict[str,]):
        super().__init__(data)
        self.id = self.raw["id"]
        self.realtime_start = self.raw["realtime_start"]
        self.realtime_end = self.raw["realtime_end"]
        self.title = self.raw["title"]
        self.observation_start = self.raw["observation_start"]
        self.observation_end = self.raw["observation_end"]
        self.frequency = self.raw["frequency"]
        self.frequency_short = self.raw["frequency_short"]
        self.units = self.raw["units"]
        self.units_short = self.raw["units_short"]
        self.seasonal_adjustment = self.raw["seasonal_adjustment"]
        self.seasonal_adjustment_short = self.raw["seasonal_adjustment_short"]
        self.last_updated = self.raw["last_updated"]
        self.popularity = self.raw["popularity"]
        self.notes = self.raw["notes"]

    def _get_all_observations_from_fred(self):
        probe = self.client.series(
            "observations", series_id=self.id, limit=1
        )  # get the first observation to get the count
        OFFSET = 0
        all_observations = [None] * probe["count"]  # preallocate the list
        while OFFSET < probe["count"]:
            response = self._get_raw_observations_from_fred(offset=OFFSET)
            all_observations[OFFSET : OFFSET + len(response["observations"])] = (
                response["observations"]
            )
            OFFSET += len(response["observations"])
        self._update_observations(all_observations)

    def _get_raw_observations_from_fred(self, **kwargs):
        return self.client.series("observations", series_id=self.id, **kwargs)

    def _update_observations(self, raw_observations: list[dict[str,]]) -> None:
        if all(raw_observations):  # if all observations are not None
            self.observations = [
                self.create_fred_observation(observation)
                for observation in raw_observations
            ]  # perhaps this should be a dictionary with date as key.
            # Problem with dictionary is that depending on the other api parameters like frequency, aggregation_method, etc, the same date might have two different values.  # noqa: E501
        else:
            self.observations = None

    def get_observations_from_fred(self, **kwargs) -> None:
        response = self._get_raw_observations_from_fred(**kwargs)
        self._update_observations(response["observations"])

    @staticmethod
    def create_fred_observation(data):
        return FredObservation._create(data)


class FredObservation(FredData):
    def __init__(self, data: dict[str,]):
        raise RuntimeError(
            "FredObservation should not be instantiated directly. Use FredSeries.get_observations_from_fred() instead"  # noqa: E501
        )

    @classmethod
    def _create(cls, data: dict[str,]):
        instance = cls.__new__(cls)
        super(FredObservation, instance).__init__(data)
        instance.realtime_start = instance.raw["realtime_start"]
        instance.realtime_end = instance.raw["realtime_end"]
        instance.date = instance.raw["date"]
        instance.value = instance.raw["value"]
        # instance.units = instance.raw["units"] # needs its own method to populate this
        # instance.fred_series_id = instance.raw["fred_series_id"] # needs its own method to populate this  # noqa: E501
        return instance


class FredResponse:
    def __init__(self, data: dict[str,], unpack_keyword: str | None = None):
        self.raw = data
        self.unpack_keyword = (
            unpack_keyword if unpack_keyword else self._infer_unpack_keyword()
        )
        self._get_metadata()
        self._get_payload()
        self.parsed_payload = self._parse_payload()

    def _get_metadata(self):
        self.metadata = {}
        for key, value in self.raw.items():
            if key != self.unpack_keyword:
                self.metadata[key] = value

    def _get_payload(self):
        self.payload = self.raw[self.unpack_keyword]

    def _infer_unpack_keyword(self):
        if "seriess" in self.raw:
            return "seriess"
        elif "categories" in self.raw:
            return "categories"
        elif "tags" in self.raw:
            return "tags"
        elif "releases" in self.raw:
            return "releases"
        elif "observations" in self.raw:
            return "observations"
        else:
            raise ValueError("Could not infer unpack keyword.")

    def _parse_payload(self) -> Generator[FredTag | FredSeries, None, None]:
        for item in self.payload:
            if self.unpack_keyword == "tags":
                yield FredTag(item)
            elif self.unpack_keyword == "seriess":
                yield FredSeries(item)
