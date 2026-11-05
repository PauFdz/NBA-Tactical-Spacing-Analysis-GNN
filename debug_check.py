import pyarrow.feather as feather
import os

path = "/mnt/work/datasets/sports_analytics/basketball/data-00000-of-00098.arrow"

try:
    print(f"Attempting to read: {path}")
    table = feather.read_table(path)
    print("Success! File is a valid Arrow/Feather file.")
    print(f"Number of rows: {table.num_rows}")
    print(f"Columns: {table.column_names}")
    # Print the first row to see structure
    print(f"First row: {table.to_pylist()[0]}")
except Exception as e:
    print(f"Failed to read as Feather: {e}")