from json import load, dump
from datetime import datetime
from datafetcher.Congress.congress.core import Congress
from datafetcher.compile_congress_training_dataset import (
    get_list_of_bills,
    download_public_law_after_date,
    download_public_law
)

# Arguments for script
requests_error_log = []
search_limit = datetime(1947, 1, 1, 00, 00, 00)
error_requests_filepath = "error_requests.json"
error_billtext_filepath = "error_bill_text.json"

with open(error_requests_filepath, 'r') as file:
    error_requests = load(file)
no_of_request_errors = len(error_requests)
with open(error_billtext_filepath, 'r') as file:
    billtext_error_log = load(file)

count = 0
old_count = 0
client = Congress()
for request in error_requests:
    count += 1
    if ((count-old_count)/no_of_request_errors) * 100 > 5:
        old_count = count
        pct_done = round((count/no_of_request_errors) * 100)
        print(f"Retrying request errors...{pct_done}% done")

    OFFSET = request["offset"]
    LIMIT = request["limit"]
    list_of_bills, errored_out = get_list_of_bills(client, OFFSET, LIMIT, requests_error_log)

    if errored_out:
        continue

    for bill in list_of_bills['bills']:
        finished_searching = download_public_law_after_date(
            search_limit,
            bill,
            client,
            billtext_error_log
        )
        if finished_searching:
            break

print("Length of requests_error_log before retrying:", no_of_request_errors)
print("Length of requests_error_log after retrying:", len(requests_error_log))

# Tackle billtext error log
no_of_billtext_errors = len(billtext_error_log)
new_billtext_error_log = []

count = 0
old_count = 0
for item in billtext_error_log:
    count += 1
    if ((count-old_count)/no_of_billtext_errors) * 100 > 5:
        old_count = count
        pct_done = round((count/no_of_billtext_errors) * 100)
        print(f"Retrying billtext errors...{pct_done}% done")

    download_public_law(item["bill"], client, new_billtext_error_log)

print("Length of billtext_error_log before retrying:", no_of_billtext_errors)
print("Length of billtext_error_log after retrying:", len(new_billtext_error_log))

if requests_error_log:
    with open("error_requests.json", 'w') as file:
        dump(requests_error_log, file)
if new_billtext_error_log:
    with open("error_bill_text.json", 'w') as file:
        dump(new_billtext_error_log, file)
