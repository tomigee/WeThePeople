import os
from json import load

import requests
import pandas as pd

from datafetcher.utils import format_source
from .fred_master.fred.core import Fred

class DataFetcher:
    config_path = os.path.join(os.path.dirname(__file__), "config.json")

    def __init__(self, source):
        with open(self.config_path, "r") as file:
            config_file = load(file)
        if not isinstance(source, str):
            raise TypeError("Source must be of data type string")
        if source not in config_file:
            raise ValueError(f"Source {source} does not exist or is not implemented")
        self._source = format_source(source)

    @property
    def source(self):
        return self._source

    def fetch_tokens(self):
        """
        Function that fetches tokens (generally API tokens) and other things needed to access data sources
        """

        with open(self.config_path, "r") as file:
            config_file = load(file)

        token = config_file[self.source]["api_token"]
        token_parameter_name = list(token.items())[0][0]
        token_parameter_value = list(token.items())[0][1]
        return token_parameter_name, token_parameter_value
    
    def append_token(self, kwargs_dict):
        
        token_param_name, token_param_value = self.fetch_tokens()
        if token_param_name not in kwargs_dict:
            kwargs_dict[token_param_name] = token_param_value
    
    def fetch_origin(self):
        
        with open(self.config_path, "r") as file:
            config_file = load(file)
        
        origin = config_file[self.source]["origin"]
        return origin
    
    def unpack_json(self, data):
        unpacked_data = pd.DataFrame()
        row_count = 0
        for row in data:
            for key, value in row.items():
                unpacked_data.loc[row_count,key] = value
            row_count += 1

        return unpacked_data
        
class BeaDataFetcher(DataFetcher):
    RESULT_FORMAT_CONST = "ResultFormat"
    METHOD_CONST = "method"
    JSON_OBJ_FIRST_INDEX = "BEAAPI"
    JSON_OBJ_SECOND_INDEX = "Results"
    
    def __init__(self):
        super().__init__("bea")
        
    def get_data(self, **kwargs):
        """
        Get data from source and organize into a dataframe
        """
        data = self.bea_request("GetData", kwargs, "Data")
        return data
    
    def get_dataset_list(self, **kwargs):
        data = self.bea_request("GETDATASETLIST", kwargs)
        return data
    
    def get_parameter_list(self, datasetname, **kwargs):
        kwargs["datasetname"] = datasetname
        data = self.bea_request("getparameterlist", kwargs, "Parameter")
        return data
        
    def bea_request(self, method, kwargs, indexer="Dataset"):
        if not isinstance(kwargs, dict):
            raise TypeError("kwargs must be a dictionary")
        
        self.append_token(kwargs)
        kwargs[self.METHOD_CONST] = method
        if self.RESULT_FORMAT_CONST not in kwargs:
            kwargs[self.RESULT_FORMAT_CONST] = "JSON"
        origin = self.fetch_origin()
        r = requests.get(origin, kwargs)
        r = r.json()
        json_data = r[self.JSON_OBJ_FIRST_INDEX][self.JSON_OBJ_SECOND_INDEX][indexer]
        data = self.unpack_json(json_data)
        return data
    
class FredDataFetcher(DataFetcher):
    def __init__(self):
        super().__init__("fred")
        _, api_key = self.fetch_tokens()
        self.fred = Fred(api_key=api_key)
    
    def get_data(self, series_id):
        """Implementation of FRED API endpoint 'fred/series/observations'

        Args:
            series_id (str): FRED Series ID

        Returns:
            pandas.DataFrame: The data values as requested
        """
        
        data = self.fred.series('observations', series_id=series_id)
        data = data["observations"]
        unpacked_data = self.unpack_json(data)
        return unpacked_data
    
    def get_dataset_info(self, series_id):
        """Implementation of FRED API endpoint 'fred/series'

        Args:
            series_id (str): FRED Series ID

        Returns:
            pandas.DataFrame: The data values as requested
        """
        
        data = self.fred.series(series_id=series_id)
        data = data["seriess"]
        unpacked_data = self.unpack_json(data)
        return unpacked_data
    
    def search_datasets(self, search_text):
        data = self.fred.series('search', search_text=search_text)
        data = data["seriess"]
        unpacked_data = self.unpack_json(data)
        return unpacked_data
    


