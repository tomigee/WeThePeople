from unittest import TestCase, mock
from copy import copy
from json import loads, load, dumps

from bea import bea
from bea.bea import Bea

# Testing methodology
# 1. Develop unit tests for each method
# 2. Develop integration tests for the public methods
# 3. Develop integration tests for the public methods, including the API's responses


# Common setup for most private functions
def common_setup(func):
    def wrapper(self, *args, **kwargs):
        func(self, *args, **kwargs)
        patcher1 = mock.patch.dict(bea.os.environ, {"BEA_API_KEY": "ABCD-EFGH-IJKL-MNOP-1234"})
        self.addClassCleanup(patcher1.stop)
        self.mock_os = patcher1.start()
        self.client = Bea()

    return wrapper


# Utilities for parsing and privatizing API responses
def strip_times_from_api_response(response):
    # Strip UTCProductionTime
    if isinstance(response, str):
        response = loads(response)
    if isinstance(response["BEAAPI"]["Results"], list):
        response["BEAAPI"]["Results"][0]["UTCProductionTime"] = "null"
    else:  # assuming dict
        response["BEAAPI"]["Results"]["UTCProductionTime"] = "null"

    # Strip LastRevised from notes
    if "Notes" in response["BEAAPI"]["Results"]:
        response["BEAAPI"]["Results"]["Notes"] = "null"

    return dumps(response)


def impute_api_token(response, api_token):
    if isinstance(response, str):
        response = loads(response)
    params = response["BEAAPI"]["Request"]["RequestParam"]
    for name_val_pair in params:
        if name_val_pair["ParameterName"] == "USERID":
            name_val_pair["ParameterValue"] = api_token
            break
    return dumps(response)


def snakecase_to_camelcase(word):
    return "".join(map(str.capitalize, word.split("_")))


# Common setup for user exposed functions that access api endpoints
def api_endpt_fn_test_class_setup(self, api_endpoint_fn_name):
    self.client = Bea()
    with open("bea/test_cases_api_responses.json", 'r') as file:
        test_data = load(file)
    self.inputs = test_data[api_endpoint_fn_name]["inputs"]
    self.responses = test_data[api_endpoint_fn_name]["responses"]


def api_endpt_fn_calls_process_request_correctly(self, api_endpt_fn, reqd_fn_args):
    patcher1 = mock.patch('requests.Session.get', autospec=True)
    self.addCleanup(patcher1.stop)
    self.mock_request = patcher1.start()

    # Discern which api_endpoint_fn is being called
    if api_endpt_fn == self.client.nipa:
        datasetname = 'NIPA'
    elif api_endpt_fn == self.client.ni_underlying_detail:
        datasetname = 'NIUnderlyingDetail'
    elif api_endpt_fn == self.client.fixed_assets:
        datasetname = 'FixedAssets'
    elif api_endpt_fn == self.client.mne_di:
        datasetname = 'MNE'
    elif api_endpt_fn == self.client.mne_amne:
        datasetname = 'MNE'
    elif api_endpt_fn == self.client.gdp_by_industry:
        datasetname = 'GDPbyIndustry'
    elif api_endpt_fn == self.client.ita:
        datasetname = 'ITA'
    elif api_endpt_fn == self.client.iip:
        datasetname = 'IIP'
    elif api_endpt_fn == self.client.input_output:
        datasetname = 'InputOutput'
    elif api_endpt_fn == self.client.underlying_gdp_by_industry:
        datasetname = 'UnderlyingGDPbyIndustry'
    elif api_endpt_fn == self.client.intl_serv_trade:
        datasetname = 'IntlServTrade'
    elif api_endpt_fn == self.client.regional:
        datasetname = 'Regional'
    elif api_endpt_fn == self.client.intl_serv_sta:
        datasetname = 'IntlServSTA'

    for input_name in self.inputs:
        with mock.patch(
            'bea.bea.Bea._Bea__process_request',
            wraps=self.client._Bea__process_request
        ) as mock_process:
            api_endpt_fn_args = self.inputs[input_name]
            api_endpt_fn(**api_endpt_fn_args)

            snakecase_args = []
            temp_dict = copy(api_endpt_fn_args)
            for key in self.inputs[input_name]:
                if key in reqd_fn_args:
                    snakecase_args.append(key)
                    temp_dict.update({snakecase_to_camelcase(key): temp_dict[key]})
            [temp_dict.pop(arg) for arg in snakecase_args]
            mock_process.assert_called_once_with(datasetname, temp_dict)


def api_endpt_fn_test_output_value(self, api_endpt_fn):
    outputs = {}
    count = 0
    for response in self.responses:
        count += 1
        outputs.update(
            {
                f'output{count}': impute_api_token(
                    strip_times_from_api_response(self.responses[response]),
                    self.client._Bea__api_token
                )
            }
        )

    for input, output in zip(self.inputs, outputs):
        self.assertEqual(
            strip_times_from_api_response(
                api_endpt_fn(**self.inputs[input])
            ),
            outputs[output]
        )


class TestBeaInit(TestCase):

    @classmethod
    @common_setup
    def setUpClass(self):
        patcher2 = mock.patch('requests.Session', autospec=True)
        self.addClassCleanup(patcher2.stop)
        self.mock_request = patcher2.start()

    # Unit tests

    def test_assigns_api_token(self):
        assert hasattr(self.client, "_Bea__api_token")
        self.assertEqual(self.client._Bea__api_token, "ABCD-EFGH-IJKL-MNOP-1234")

    def test_assigns_query_params(self):
        assert hasattr(self.client, "_Bea__query_params")
        # TODO: Should I test for the different entries in __query_params? Yes, you should

    def test_assigns_origin_url(self):
        assert hasattr(self.client, "_Bea__origin_url")

    def test_output_value_origin_url(self):
        self.assertEqual(
            self.client._Bea__origin_url,
            "https://apps.bea.gov/api/data"
        )

    def test_initiates_requests_session(self):
        assert hasattr(self.client, "request_session")
        self.mock_request.assert_called()

    # TODO: Do I have to implement tearDownClass()?
    @classmethod
    def tearDownClass(self):
        del self.client


class TestBeaValidateInputs(TestCase):

    @classmethod
    @common_setup
    def setUpClass(self):
        pass

    # Unit tests
    def test_output_value(self):
        params_test_cases = {
            "input1": {
                "dummy_key1": 1234,
                "dummy_key2": "dummy_val"
                },
            "output1": {
                "UserID": self.client._Bea__api_token,
                "method": "GetData",
                "ResultFormat": "json",
                "dummy_key1": 1234,
                "dummy_key2": "dummy_val"
                },
            "input2": {
                "dummy_key1": "dummy_val1",
                "method": "DummyMethod",
                "ResultFormat": "xml"
                },
            "output2": {
                "UserID": self.client._Bea__api_token,
                "dummy_key1": "dummy_val1",
                "method": "DummyMethod",
                "ResultFormat": "xml"
            }
        }

        self.assertEqual(self.client._Bea__validate_inputs(), self.client._Bea__query_params)
        self.assertEqual(
            self.client._Bea__validate_inputs(params_test_cases["input1"]),
            params_test_cases["output1"]
        )
        self.assertEqual(
            self.client._Bea__validate_inputs(params_test_cases["input2"]),
            params_test_cases["output2"]
        )

    @classmethod
    def tearDownClass(self):
        del self.client


class TestBeaComposeFullUrl(TestCase):

    @classmethod
    @common_setup
    def setUpClass(self):
        pass

    # Unit tests
    def test_output_value(self):
        self.assertEqual(
            self.client._Bea__compose_full_url(),
            self.client._Bea__origin_url
        )
        self.assertEqual(
            self.client._Bea__compose_full_url("/this/url/should/not/be/concatenated"),
            self.client._Bea__origin_url
        )

    @classmethod
    def tearDownClass(self):
        del self.client


class TestSendRequest(TestCase):

    @classmethod
    @common_setup
    def setUpClass(self):
        patcher2 = mock.patch('requests.Session.get', autospec=True)
        self.addClassCleanup(patcher2.stop)
        self.mock_request = patcher2.start()

    # Unit tests
    def test_request_made_with_correct_arguments(self):
        test_cases = {
            "input1": (
                "https://apps.bea.gov/api/data",
                {
                    "key1": "val1",
                    "key2": "val2"
                }
            )
        }
        self.client._Bea__send_request(*test_cases["input1"])
        self.mock_request.assert_called_once_with(
            self.client.request_session,
            test_cases["input1"][0],
            params=test_cases["input1"][1]
        )

    @classmethod
    def tearDownClass(self):
        del self.client


class TestProcessRequest(TestCase):

    @classmethod
    @common_setup
    def setUpClass(self):
        pass

    # Unit tests
    def test_calls_validate_inputs_correctly(self):
        test_cases = {
            "input1": (
                "NIPA",
                {
                    "key1": "val1",
                    "key2": "val2"
                }
            ),
            "output1": {
                "key1": "val1",
                "key2": "val2",
                "datasetname": "NIPA"
            }
        }

        with mock.patch(
            'bea.bea.Bea._Bea__validate_inputs',
            wraps=self.client._Bea__validate_inputs
        ) as small_mock:
            self.client._Bea__process_request(*test_cases["input1"])
            small_mock.assert_called_once_with(
                test_cases["output1"]
            )

    def test_calls_compose_full_url_correctly(self):
        test_cases = {
            "input1": (
                "NIPA",
                {
                    "key1": "val1",
                    "key2": "val2"
                }
            ),
        }

        with mock.patch(
            'bea.bea.Bea._Bea__compose_full_url',
            wraps=self.client._Bea__compose_full_url
        ) as small_mock:
            self.client._Bea__process_request(*test_cases["input1"])
            calls = [mock.call()]
            small_mock.assert_has_calls(calls, any_order=True)

    def test_calls_send_request_correctly(self):
        test_inputs = {
            "input1": (
                "NIPA",
                {
                    "key1": "val1",
                    "key2": "val2"
                }
            ),
        }

        test_outputs = {
            "output1": (
                self.client._Bea__compose_full_url(),
                self.client._Bea__validate_inputs(
                    {
                        **test_inputs["input1"][1],
                        "datasetname": test_inputs["input1"][0]
                    }
                )
            ),
        }

        with mock.patch(
            'bea.bea.Bea._Bea__send_request',
            wraps=self.client._Bea__send_request
        ) as small_mock:
            self.client._Bea__process_request(*test_inputs["input1"])
            small_mock.assert_called_once_with(
                *test_outputs["output1"]
            )

    # Integration tests w/ API
    def test_output_value(self):
        pass

    @classmethod
    def tearDownClass(self):
        del self.client


class TestNipa(TestCase):

    @classmethod
    def setUpClass(self):
        api_endpt_fn_test_class_setup(self, "nipa")

    # Unit tests
    def test_calls_process_request_correctly(self):
        reqd_args = ["year", "frequency", "table_name"]
        api_endpt_fn_calls_process_request_correctly(self, self.client.nipa, reqd_args)

    # Integration tests w API
    def test_output_value_API(self):
        api_endpt_fn_test_output_value(self, self.client.nipa)

    @classmethod
    def tearDownClass(self):
        del self.client


class TestNiUnderlyingDetail(TestCase):

    @classmethod
    def setUpClass(self):
        api_endpt_fn_test_class_setup(self, "ni_underlying_detail")

    # Unit tests
    def test_calls_process_request_correctly(self):
        reqd_args = ["year", "frequency", "table_name"]
        api_endpt_fn_calls_process_request_correctly(
            self, self.client.ni_underlying_detail,
            reqd_args
        )

    # Integration tests w API
    def test_output_value_API(self):
        api_endpt_fn_test_output_value(self, self.client.ni_underlying_detail)

    @classmethod
    def tearDownClass(self):
        del self.client


class TestFixedAssets(TestCase):

    @classmethod
    def setUpClass(self):
        api_endpt_fn_test_class_setup(self, "fixed_assets")

    # Unit tests
    def test_calls_process_request_correctly(self):
        reqd_args = ["year", "table_name"]
        api_endpt_fn_calls_process_request_correctly(
            self, self.client.fixed_assets, reqd_args
        )

    # Integration tests w API
    def test_output_value_API(self):
        api_endpt_fn_test_output_value(self, self.client.fixed_assets)

    @classmethod
    def tearDownClass(self):
        del self.client


class TestMneDi(TestCase):

    @classmethod
    def setUpClass(self):
        api_endpt_fn_test_class_setup(self, "mne_di")

    # Unit tests
    def test_calls_process_request_correctly(self):
        reqd_args = ["year", "direction_of_investment", "classification"]
        api_endpt_fn_calls_process_request_correctly(
            self, self.client.mne_di,
            reqd_args
        )

    # Integration tests w API
    def test_output_value_API(self):
        api_endpt_fn_test_output_value(self, self.client.mne_di)

    @classmethod
    def tearDownClass(self):
        del self.client


class TestMneAmne(TestCase):

    @classmethod
    def setUpClass(self):
        api_endpt_fn_test_class_setup(self, "mne_amne")

    # Unit tests
    def test_calls_process_request_correctly(self):
        reqd_args = ["year",
                     "direction_of_investment",
                     "classification",
                     "ownership_level",
                     "non_bank_affiliates_only"]
        api_endpt_fn_calls_process_request_correctly(
            self, self.client.mne_amne,
            reqd_args
        )

    # Integration tests w API
    def test_output_value_API(self):
        api_endpt_fn_test_output_value(self, self.client.mne_amne)

    @classmethod
    def tearDownClass(self):
        del self.client


class TestGdpByIndustry(TestCase):

    @classmethod
    def setUpClass(self):
        api_endpt_fn_test_class_setup(self, "gdp_by_industry")

    # Unit tests
    def test_calls_process_request_correctly(self):
        reqd_args = ["year", "frequency", "table_id", "industry"]
        api_endpt_fn_calls_process_request_correctly(
            self, self.client.gdp_by_industry,
            reqd_args
        )

    # Integration tests w API
    def test_output_value_API(self):
        api_endpt_fn_test_output_value(self, self.client.gdp_by_industry)

    @classmethod
    def tearDownClass(self):
        del self.client


class TestIta(TestCase):

    @classmethod
    def setUpClass(self):
        api_endpt_fn_test_class_setup(self, "ita")

    # Unit tests
    def test_calls_process_request_correctly(self):
        reqd_args = ["indicator", "area_or_country"]
        api_endpt_fn_calls_process_request_correctly(
            self, self.client.ita, reqd_args
        )

    # Integration tests w API
    def test_output_value_API(self):
        api_endpt_fn_test_output_value(self, self.client.ita)

    @classmethod
    def tearDownClass(self):
        del self.client


class TestIip(TestCase):

    @classmethod
    def setUpClass(self):
        api_endpt_fn_test_class_setup(self, "iip")

    # Unit tests
    def test_calls_process_request_correctly(self):
        reqd_args = ["year", "type_of_investment"]
        api_endpt_fn_calls_process_request_correctly(
            self, self.client.iip, reqd_args
        )

    # Integration tests w API
    def test_output_value_API(self):
        api_endpt_fn_test_output_value(self, self.client.iip)

    @classmethod
    def tearDownClass(self):
        del self.client


class TestInputOutput(TestCase):

    @classmethod
    def setUpClass(self):
        api_endpt_fn_test_class_setup(self, "input_output")

    # Unit tests
    def test_calls_process_request_correctly(self):
        reqd_args = ["year", "table_id"]
        api_endpt_fn_calls_process_request_correctly(
            self, self.client.input_output,
            reqd_args
        )

    # Integration tests w API
    def test_output_value_API(self):
        api_endpt_fn_test_output_value(self, self.client.input_output)

    @classmethod
    def tearDownClass(self):
        del self.client


class TestUnderlyingGdpByIndustry(TestCase):

    @classmethod
    def setUpClass(self):
        api_endpt_fn_test_class_setup(self, "underlying_gdp_by_industry")

    # Unit tests
    def test_calls_process_request_correctly(self):
        reqd_args = ["year", "frequency", "table_id", "industry"]
        api_endpt_fn_calls_process_request_correctly(
            self, self.client.underlying_gdp_by_industry,
            reqd_args
        )

    # Integration tests w API
    def test_output_value_API(self):
        api_endpt_fn_test_output_value(self, self.client.underlying_gdp_by_industry)

    @classmethod
    def tearDownClass(self):
        del self.client


class TestIntlServTrade(TestCase):

    @classmethod
    def setUpClass(self):
        api_endpt_fn_test_class_setup(self, "intl_serv_trade")

    # Unit tests
    def test_calls_process_request_correctly(self):
        reqd_args = ["type_of_service", "area_or_country"]
        api_endpt_fn_calls_process_request_correctly(
            self, self.client.intl_serv_trade,
            reqd_args
        )

    # Integration tests w API
    def test_output_value_API(self):
        api_endpt_fn_test_output_value(self, self.client.intl_serv_trade)

    @classmethod
    def tearDownClass(self):
        del self.client


class TestRegional(TestCase):

    @classmethod
    def setUpClass(self):
        api_endpt_fn_test_class_setup(self, "regional")

    # Unit tests
    def test_calls_process_request_correctly(self):
        reqd_args = ["table_name", "line_code", "geo_fips"]
        api_endpt_fn_calls_process_request_correctly(
            self, self.client.regional,
            reqd_args
        )

    # Integration tests w API
    def test_output_value_API(self):
        api_endpt_fn_test_output_value(self, self.client.regional)

    @classmethod
    def tearDownClass(self):
        del self.client


class TestIntlServSta(TestCase):

    @classmethod
    def setUpClass(self):
        api_endpt_fn_test_class_setup(self, "intl_serv_sta")

    # Unit tests
    def test_calls_process_request_correctly(self):
        reqd_args = []
        api_endpt_fn_calls_process_request_correctly(
            self, self.client.intl_serv_sta,
            reqd_args
        )

    # Integration tests w API
    def test_output_value_API(self):
        api_endpt_fn_test_output_value(self, self.client.intl_serv_sta)

    @classmethod
    def tearDownClass(self):
        del self.client
