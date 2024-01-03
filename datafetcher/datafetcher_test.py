import unittest
import string
import random
from datafetcher import * 
from json import load

test_sources = ["bea"]
erroneous_test_sources = [1,0,-1,True,["k"],0+5j,"notfred"]
with open("config.json", "r") as config_file:
    config = load(config_file)

class TestInit(unittest.TestCase):
    """Class for unit tests of DataFetcher.__init__() function
    """
    def test_output_value(self):
        """Tests that the DataFetcher object is constructed correctly
        """
        for source in test_sources:
            my_datafetcher = DataFetcher(source)
            self.assertEqual(my_datafetcher.source, source)
        
    def test_input_types(self):
        """Tests that the DataFetcher constructor raises errors correctly when given invalid inputs
        """
        for source in erroneous_test_sources:
            if not isinstance(source, str):
                self.assertRaises(TypeError, DataFetcher, source)
            else:
                self.assertRaises(ValueError, DataFetcher, source)

class TestFetchTokens(unittest.TestCase):
    """Collection of unit tests for fetch_tokens()
    """
    def test_output_value(self):
        """
        Test the value of the output for fetch_tokens()
        """
        for source in test_sources:
            control = config[source]["api_token"] # correct answer
            my_datafetcher = DataFetcher(source)
            res = my_datafetcher.fetch_tokens() # test answer
            self.assertEqual(res, control)
            
    def test_output_value_length(self):
        """Tests the length of the output value of fetch_tokens()
        """
        for source in test_sources:
            my_datafetcher = DataFetcher(source)
            self.assertEqual(len(my_datafetcher.fetch_tokens()), 1)

    def test_output_type(self):
        """Tests the data type of the output returned from fetch_tokens()
        """
        for source in test_sources:
            my_datafetcher = DataFetcher(source)
            self.assertIsInstance(my_datafetcher.fetch_tokens(), dict)
            self.assertIsInstance(list(my_datafetcher.fetch_tokens().keys())[0], str)
            self.assertIsInstance(list(my_datafetcher.fetch_tokens().values())[0], str)
            
class TestFetchOrigin(unittest.TestCase):
    """Collection of unit tests for fetch_origin()
    """
    def test_output_value(self):
        """Tests the output value of fetch_origin()
        """
        for source in test_sources:
            control = config[source]["origin"] # correct answer
            my_datafetcher = DataFetcher(source)
            res = my_datafetcher.fetch_origin() # test answer
            self.assertEqual(res, control)
            
    def test_output_type(self):
        """Tests the data type of the output of fetch_origin()
        """
        for source in test_sources:
            my_datafetcher = DataFetcher(source)
            self.assertIsInstance(my_datafetcher.fetch_origin(), str)
                 
            
class TestUnpackJson(unittest.TestCase):
    """Collection of unit tests for unpack_json()
    """
    def test_input_types(self):
        """Tests data types of the inputs to unpack_json()
        """
        pass
    def test_output_types(self):
        """Tests data types of the outputs from unpack_json()
        """
        pass
    def test_output_values(self):
        """Tests the values of the outputs from unpack_json()
        """
        pass
    
class TestAppendToken(unittest.TestCase):
    """Collection of unit tests for append_token()
    """
    def test_output_values(self):
        """Tests the values of the outputs from append_token
        """
        
        for source in test_sources:
            my_datafetcher = DataFetcher(source)
            test_dict = {"a":1}
            self.assertEqual(my_datafetcher.append_token(test_dict), test_dict.update(config[source]["api_token"]))
    
# class TestHeaderCompiler(unittest.TestCase):
#     """Collection of unit tests for header_compiler()
#     """
#     def test_output_value(self):
#         """Tests the value output from header_compiler()
#         """
#         for source in test_sources:
#             origin = config[source]["origin"]
#             prefix = config[source]["prefix"]
#             control = origin + prefix # correct answer
            
#             my_datafetcher = DataFetcher(source)
#             res = my_datafetcher.header_compiler() # test answer
#             self.assertEqual(res, control)
            
#     def test_output_type(self):
#         """Tests the data type of the output from header_compiler()
#         """
#         for source in test_sources:
#             my_datafetcher = DataFetcher(source)
#             self.assertIsInstance(my_datafetcher.header_compiler(), str)

# class TestRequestCompiler(unittest.TestCase):
#     """Collection of unit tests for request_compiler()
#     """
#     def test_output_value(self):
#         """Tests the value output from request_compiler()
#         """
#         def id_generator(section_count, section_length=1, hyphenated=False):
#             """Creates a random string of a) section_length if hyphenated=False or b) section_length * section_count if hyphenated=True

#             Args:
#                 section_count (int): _description_
#                 section_length (int, optional): _description_
#                 hyphenated (bool, optional): _description_. Defaults to False.

#             Returns:
#                 _type_: _description_
#             """

#             if hyphenated:
#                 token = []
#                 for section in range(section_count):
#                     id_section = ''.join(random.SystemRandom().choice(string.ascii_uppercase + string.digits + string.ascii_lowercase) for _ in range(section_length))
#                     token.append(id_section)

#                 return '-'.join(token)

#             else:
#                 for section in range(section_count * section_length):
#                     token = ''.join(random.SystemRandom().choice(string.ascii_uppercase + string.digits + string.ascii_lowercase) for _ in range(section_count * section_length))

#                 return token

#         test_cases = [
#             {"source":"bea",
#              "kwargs":{
#                  "param1":id_generator(4,6),
#                  "param2":id_generator(24)
#                  }
#              },
#             {"source":"bea",
#              "kwargs":{
#                  "param1":id_generator(4,6),
#                  "param2":id_generator(24)
#                  }
#              },
#             {"source":"bea",
#              "kwargs":{
#                  "param1":id_generator(4,6),
#                  "param2":id_generator(24)
#                  }
#              }
#             ]
#         for case in test_cases:
#             my_datafetcher = DataFetcher(case["source"])
#             kwargs = case["kwargs"]
#             res = my_datafetcher.request_compiler(**kwargs)
            
#             # Create control
#             origin = config[case["source"]]["origin"]
#             prefix = config[case["source"]]["prefix"]
#             sep = config[case["source"]]["sep"]
#             token_dict = config[case["source"]]["api_token"]
#             header = origin + prefix
#             rq_elements = [list(token_dict.items())[0][0] + "=" + list(token_dict.items())[0][1]]
#             for key, value in kwargs.items():
#                 temp = key + "=" + value
#                 rq_elements.append(temp)
#             body = sep.join(rq_elements)
#             control_rq = header + body
#             self.assertEqual(res, control_rq)

# class TestFetchPrefix(unittest.TestCase):
#     """Collection of unit tests for fetch_prefix()
#     """
#     def test_output_value(self):
#         """Tests the value of the output of fetch_prefix()
#         """
#         for source in test_sources:
#             control = config[source]["prefix"] # correct answer
#             my_datafetcher = DataFetcher(source)
#             res = my_datafetcher.fetch_prefix() # test answer
#             self.assertEqual(res, control)
            
#     def test_output_type(self):
#         """Tests the data type of the output of fetch_prefix()
#         """
#         for source in test_sources:
#             my_datafetcher = DataFetcher(source)
#             self.assertIsInstance(my_datafetcher.fetch_prefix(), str)
            
# class TestFetchSep(unittest.TestCase):
    # """Collection of unit tests for fetch_sep()
    # """
    # def test_output_value(self):
    #     """Tests the value output from fetch_sep()
    #     """
    #     for source in test_sources:
    #         control = config[source]["sep"] # correct answer
    #         my_datafetcher = DataFetcher(source)
    #         res = my_datafetcher.fetch_sep() # test answer
    #         self.assertEqual(res, control)
            
    # def test_output_type(self):
    #     """Tests the data type of the output from fetch_sep()
    #     """
    #     for source in test_sources:
    #         my_datafetcher = DataFetcher(source)
    #         self.assertIsInstance(my_datafetcher.fetch_sep(), str)