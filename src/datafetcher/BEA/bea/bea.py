# TODO: APIError handling

import os
from json import load
from copy import copy

import requests


class Bea:
    methods = ["GetData",
               "GetDatasetList",
               "GetParameterList",
               "GetParameterValues",
               "GetParameterValuesFiltered"]

    # Load config file
    dataset_args_filename = "datasets_args.json"
    dataset_args_filepath = os.path.join(os.path.dirname(__file__), dataset_args_filename)
    with open(dataset_args_filepath, "r") as file:
        datasets_args = load(file)

    def __init__(self):
        self.__api_token = os.environ["BEA_API_KEY"]
        self.__query_params = {
            "UserID": self.__api_token,
            "method": "GetData",
            "ResultFormat": "json",
        }
        self.__origin_url = "https://apps.bea.gov/api/data"
        self.request_session = requests.Session()

# PRIVATE METHODS
    def __validate_inputs(self, params=None):
        if params is not None:
            if "datasetname" in params:  # Validate param values if we know dataset_name
                # TODO: Custom logic for MNE Dataset
                # TODO: Validate param values
                dataset_name = params["datasetname"]
                if "method" not in params:  # escapes this statement if the method is
                    # _get_parameter_values
                    if dataset_name == 'ITA':
                        if (params["Indicator"] is None) and (params["AreaOrCountry"] is None):
                            raise TypeError(
                                "Either exactly one Indicator must be requested or exactly one\
                                AreaOrCountry other than AllCountries must be requested."
                            )

                    if dataset_name == 'IIP':
                        if (params["Year"] is None) and (params["TypeOfInvestment"] is None):
                            raise TypeError(
                                "Either exactly one TypeOfInvestment must be requested\
                                or exactly one Year must be requested."
                            )

                    if dataset_name == 'IntlServTrade':
                        if (params["TypeOfService"] is None) and (params["AreaOrCountry"] is None):
                            raise TypeError(
                                "Either exactly one TypeOfService must be requested or exactly one\
                                AreaOrCountry other than AllCountries must be requested."
                            )

        # Overwrite default query params with user-supplied params
        query_params = copy(self.__query_params)
        if params is not None:
            for param in params:
                query_params[param] = params[param]
        return query_params

    def __compose_full_url(self, path=None):
        # Creating this method just in case the implementation of URLs changes in the future
        return self.__origin_url

    def __send_request(self, full_url, kwargs):
        response = self.request_session.get(full_url, params=kwargs)
        if response.ok:
            return response
        else:
            raise requests.exceptions.RequestException()

    def __process_request(self, dataset_name, params):
        params = copy(params)
        params["datasetname"] = dataset_name
        query_params = self.__validate_inputs(params)
        full_url = self.__compose_full_url()
        response = self.__send_request(full_url, query_params)
        return response

# PROTECTED METHODS
    def _get_parameter_values(self, dataset_name, parameter_name, **kwargs):
        kwargs["method"] = "GetParameterValues"
        kwargs["ParameterName"] = parameter_name
        response = self.__process_request(dataset_name, kwargs)
        return response.text

# PUBLIC METHODS
    def nipa(self, year, frequency, table_name, **kwargs):
        kwargs["Year"], kwargs["Frequency"], kwargs["TableName"] = year, frequency, table_name
        # print(kwargs)
        response = self.__process_request('NIPA', kwargs)
        return response.text

    def ni_underlying_detail(self, year, frequency, table_name, **kwargs):
        kwargs["Year"], kwargs["Frequency"], kwargs["TableName"] = year, frequency, table_name
        response = self.__process_request('NIUnderlyingDetail', kwargs)
        return response.text

    def fixed_assets(self, year, table_name, **kwargs):
        kwargs["Year"], kwargs["TableName"] = year, table_name
        response = self.__process_request('FixedAssets', kwargs)
        return response.text

    def mne_di(self, direction_of_investment, classification, year, **kwargs):
        kwargs["Year"] = year
        kwargs["DirectionOfInvestment"] = direction_of_investment
        kwargs["Classification"] = classification
        response = self.__process_request('MNE', kwargs)
        return response.text

    def mne_amne(self,
                 direction_of_investment,
                 classification,
                 year,
                 ownership_level,
                 non_bank_affiliates_only,
                 **kwargs):
        kwargs["DirectionOfInvestment"] = direction_of_investment
        kwargs["Classification"] = classification
        kwargs["Year"] = year
        kwargs["OwnershipLevel"] = ownership_level
        kwargs["NonBankAffiliatesOnly"] = non_bank_affiliates_only
        response = self.__process_request('MNE', kwargs)
        return response.text

    def gdp_by_industry(self, table_id, frequency, year, industry, **kwargs):
        kwargs["TableId"] = table_id
        kwargs["Frequency"] = frequency
        kwargs["Year"] = year
        kwargs["Industry"] = industry
        response = self.__process_request('GDPbyIndustry', kwargs)
        return response.text

    def ita(self, indicator=None, area_or_country=None, **kwargs):
        kwargs["Indicator"] = indicator
        kwargs["AreaOrCountry"] = area_or_country
        response = self.__process_request('ITA', kwargs)
        return response.text

    def iip(self, year=None, type_of_investment=None, **kwargs):
        kwargs["Year"] = year
        kwargs["TypeOfInvestment"] = type_of_investment
        response = self.__process_request('IIP', kwargs)
        return response.text

    def input_output(self, table_id, year, **kwargs):
        kwargs["TableId"], kwargs["Year"] = table_id, year
        response = self.__process_request('InputOutput', kwargs)
        return response.json()

    def underlying_gdp_by_industry(self, table_id, frequency, year, industry, **kwargs):
        kwargs["TableId"] = table_id
        kwargs["Frequency"] = frequency
        kwargs["Year"] = year
        kwargs["Industry"] = industry
        response = self.__process_request('UnderlyingGDPbyIndustry', kwargs)
        return response.text

    def intl_serv_trade(self, type_of_service=None, area_or_country=None, **kwargs):
        kwargs["TypeOfService"] = type_of_service
        kwargs["AreaOrCountry"] = area_or_country
        response = self.__process_request('IntlServTrade', kwargs)
        return response.text

    def regional(self, table_name, line_code, geo_fips, **kwargs):
        kwargs["TableName"] = table_name
        kwargs["LineCode"] = line_code
        kwargs["GeoFips"] = geo_fips
        response = self.__process_request('Regional', kwargs)
        return response.text

    def intl_serv_sta(self, **kwargs):
        response = self.__process_request('IntlServSTA', kwargs)
        return response.text
