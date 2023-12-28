from json import load
from utils import format_source

class DataFetcher:

    def __init__(self, source):
        if not isinstance(source, str):
            raise TypeError("Source must be of data type string")
        self._source = format_source(source)

    @property
    def source(self):
        return self._source

    def fetch_tokens(self):
        """
        Function that fetches tokens (generally API tokens) and other things needed to access data sources
        """

        with open("config.json", "r") as file:
            config_file = load(file)

        token = config_file[self.source]["api_token"]
        return token
    
    def fetch_origin(self):
        
        with open("config.json", "r") as file:
            config_file = load(file)
        
        origin = config_file[self.source]["origin"]
        return origin
    
    def fetch_prefix(self):
        
        with open("config.json", "r") as file:
            config_file = load(file)
        
        prefix = config_file[self.source]["prefix"]
        return prefix
    
    def fetch_sep(self):
        
        with open("config.json", "r") as file:
            config_file = load(file)
        
        sep = config_file[self.source]["sep"]
        return sep
    
    def header_compiler(self):
        
        api_origin = self.fetch_origin()
        prefix = self.fetch_prefix()
        header = api_origin + prefix
        return header

    def get_data(self, **kwargs):
        """
        Get data from source
        """

        with open("config.json", "r") as file:
            config_file = load(file)

        # Error checking
        if self.source not in config_file: # Ensure that source is implemented
            raise NotImplementedError('Data source ' + self.source + ' has not been implemented yet.')

        if self.source == "bea":
            kwargs["method"] = "GetData"
            rq = self.request_compiler(**kwargs)

    def request_compiler(self, **kwargs):
        """
        Compiles all parameters needed for a valid API request from source, errors out if invalid or incomplete parameters are provided

        kwargs are passed directly to the request string
        """

        # Error checking


        if self.source == "bea":
            # Parse kwargs to make sure the request is valid

            api_token = self.fetch_tokens()
            sep = self.fetch_sep()
            header = self.header_compiler()

            # Assemble request arguments into a list for downstream formatting
            rq_args = [list(api_token.items())[0][0] + "=" + list(api_token.items())[0][1]]
            for key, value in kwargs.items():
                rq_args.append(key + "=" + value)
            body = sep.join(rq_args) # format request arguments
            rq = header + body
            return rq


