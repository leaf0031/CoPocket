import yaml

# Update the cleaned SMILES based on the pocket data in the sequence.


with open('./pocket_sequence_test.yaml', 'r') as file1:
    data1 = yaml.full_load(file1)


with open('./clean_test.yaml', 'r') as file2:
    data2 = yaml.full_load(file2)


keys_to_keep = set(data1.keys())


filtered_data2 = {key: data2[key] for key in keys_to_keep if key in data2}


with open('filter_sequence_test.yaml', 'w') as file2:
    yaml.dump(filtered_data2, file2)

print("Update complete, retained keys:", filtered_data2.keys())
