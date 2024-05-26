from io import StringIO
from json import dump
from html.parser import HTMLParser


class MLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.reset()
        self.strict = False
        self.convert_charrefs = True
        self.text = StringIO()

    def handle_data(self, d):
        self.text.write(d)

    def get_data(self):
        return self.text.getvalue()


def strip_tags(html: str) -> str:
    s = MLStripper()
    s.feed(html)
    return s.get_data()


def process_filename(filename: str) -> str:
    """
    Removes forbidden characters "/" and ":" from filename

    Args:
        filename (str): Filename to search for forbidden characters

    Returns:
        file_name (str): Filename with forbidden characters replaced by whitespace
    """
    file_name = filename
    forbidden_chars = ["/", ":"]
    for char in forbidden_chars:
        file_name = file_name.replace(char, " ")
    return file_name


def truncate_filepath(file_path: str) -> str:
    from os import path

    # Check the filename length
    file_name = path.basename(file_path)
    starting_length = len(file_name)
    max_len = 255
    min_chars_to_drop_1 = starting_length - max_len

    # Check the pathname length
    starting_length = len(file_path)
    max_len = 1024
    min_chars_to_drop_2 = starting_length - max_len

    min_chars_to_drop = max(
        min_chars_to_drop_1,
        min_chars_to_drop_2
    )

    # Drop characters from the filename in order to meet requirements
    if min_chars_to_drop > 0:
        filename_words = file_name.split(sep=" ")
        char_count = 0
        i = 0
        for word in filename_words:
            i += 1
            char_count += len(word) + 1  # 1 for the delimiter
            if char_count >= min_chars_to_drop:
                break
        filename_words = filename_words[i:]
        new_file_name = " ".join(filename_words)
        new_file_path = path.join(
            path.dirname(file_path),
            new_file_name
        )
        return new_file_path
    else:
        return file_path


def json_to_csv(json_structured_obj, filepath: str, memory_efficient: bool = False) -> None:
    from json import dumps
    from io import StringIO

    import pandas as pd

    if not memory_efficient:
        if not isinstance(json_structured_obj, str):
            new_obj = StringIO(dumps(json_structured_obj))
        else:
            new_obj = StringIO(json_structured_obj)
        pd.read_json(new_obj).to_csv(filepath, index=False)
    else:
        all_columns = set()
        for entry in json_structured_obj:
            for column in entry:
                if column not in all_columns:
                    all_columns.add(column)

        all_columns = list(all_columns)
        with open(filepath, 'a') as file:
            file.write(",".join(all_columns) + "\n")

        for entry in json_structured_obj:
            csv_entry = [f"{entry[column]}" if column in entry else "" for column in all_columns]
            str_to_write = ",".join(csv_entry)
            with open(filepath, 'a') as file:
                file.write(str_to_write + "\n")


def write_error_files(kwargs: dict[str,]) -> None:
    """
    Writes error files to disk

    Args:
        kwargs (dict, optional): Dictionary of key, value pairs where key will become the name of
        the file and value will be the data written to the file

    Returns:
        None
    """
    for key, value in kwargs.items():
        if value:
            with open(key, 'w') as file:
                dump(value, file)


def compose_error_entry(**kwargs):
    return kwargs
