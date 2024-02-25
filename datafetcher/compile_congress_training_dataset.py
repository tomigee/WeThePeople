# This file contains a script that compiles:
# 1. All passed legislature in the United States since 1947

# CONGRESS DATA
# 1. Scrape all bill data and download it locally. Amazon S3 doesn't give me much free storage
# sadly.
# - One bill per text file
#
# -------------------------------------------------------------------------------------------
from utils import write_to_error_file


def count_leaves(tree):
    count = 0
    if isinstance(tree, list) or isinstance(tree, set):
        count += len(tree)
    else:  # assuming dictionary
        for key in tree:
            count += count_leaves(tree[key])
    return count


def get_public_law_number(phrase):
    import re
    pattern = r"Became Public Law No: [\d]+-([\d]+)."
    re_match = re.match(pattern, phrase)
    if re_match is not None:
        pl_num = re_match.group(1)
        return pl_num
    else:
        return None


def script_to_run():
    import traceback as tb
    from json import loads
    from datetime import datetime

    from .Congress.congress.core import Congress
    from .apidata import get_law_text

    client = Congress()
    offset = 0
    search_limit = datetime(1947, 1, 1, 00, 00, 00)
    latest_action_date = datetime.now()
    finished_searching = False
    while_loop_count = 0

    while not finished_searching:
        while_loop_count += 1

        # Query the Congress API
        LIMIT = 250
        response = loads(
            client.bill(
                offset=offset,
                limit=LIMIT,
                throttle=True
            )
        )

        for bill in response['bills']:
            latest_action_date = datetime.strptime(
                bill['latestAction']['actionDate'],
                "%Y-%m-%d"
            )

            if latest_action_date > search_limit:
                latest_action = bill['latestAction']['text']
                if "became public law" in latest_action.lower():
                    try:
                        congress_number = bill['congress']
                        bill_type = bill['type']
                        bill_number = bill['number']
                        law_number = get_public_law_number(latest_action)
                        bill_title = f"{bill['title']} {congress_number}_{law_number}"
                        get_law_text(
                            congress_number,
                            law_number,
                            bill_title,
                            bill_type,
                            bill_number,
                            "Public Laws"
                        )
                    except Exception:
                        # Log attributes to a text file
                        write_to_error_file(f"Exception encountered:{tb.format_exc()}", False)
                        if 'congress_number' in locals():
                            write_to_error_file(f"Congress Number: {congress_number}")
                        if 'bill_type' in locals():
                            write_to_error_file(f"Bill Type: {bill_type}")
                        if 'bill_number' in locals():
                            write_to_error_file(f"Bill Number: {bill_number}")
                        if 'law_number' in locals():
                            write_to_error_file(f"Law Number: {law_number}")
                        if 'bill_title' in locals():
                            write_to_error_file(f"Bill Title: {bill_title}")
                        write_to_error_file("\n", False)

            else:
                finished_searching = True
                break

        offset += LIMIT

        if while_loop_count % 5 == 0:
            print("Current latest action date: ", latest_action_date)
