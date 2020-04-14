import pandas as pd
from os import path


def read_data_file(file_path: str, second_column_name: str = 'count') -> pd.DataFrame:
    if not path.exists(file_path):
        raise ValueError('File not found at {}'.format(file_path))
    data = pd.read_table(file_path, delim_whitespace=True, names=['date', second_column_name], parse_dates=['date'])
    return data
