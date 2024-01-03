import json

config = {
    "bea":{
        "api_token":{
            'UserID':'441C28D2-D3CB-48DC-B794-24FED8604E8F'
            },
        "origin":'https://apps.bea.gov/api/data',
        },
    "fred":{
        "api_token":{
            'api_key':'696bb4b52535426b7760f10130dd9a31'
            },
        "origin":'https://api.stlouisfed.org/fred/series',
    }
}

with open("config.json", "w") as config_file:
    json.dump(config, config_file)