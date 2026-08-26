import csv
import yaml

# Step 1: Read the CSV file and extract the keys
csv_file = 'data_crossdocked/error_output_train.csv'
yaml_file = 'data_crossdocked/filter_sequence_train.yaml'

# Read the CSV file and extract the data
with open(csv_file, 'r') as f:
    csv_data = f.read().split(',')  # Split the file by commas
    # csv_keys = set(csv_data)  # Use a set for efficient lookup

# Step 2: Read the YAML file
with open(yaml_file, 'r') as f:
    yaml_data = yaml.safe_load(f)

# Step 3: Remove keys from the YAML data that exist in the CSV file
filtered_yaml = {key: value for key, value in yaml_data.items() if key not in csv_data}

# Step 4: Write the filtered data back to the YAML file
with open('data_crossdocked/filter_sequence_dssp_train.yaml', 'w') as f:
    yaml.dump(filtered_yaml, f, default_flow_style=False)

print('The filtered YAML file has been saved.')