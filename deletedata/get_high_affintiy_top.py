import os
import subprocess
import numpy as np
from Bio.PDB.PDBParser import PDBParser
import warnings
import yaml
import glob
from rdkit import Chem
from rdkit.Chem.rdMolAlign import CalcRMS
from easydict import EasyDict
import json
import re
import csv
from collections import defaultdict
import statistics

config_dir= './evaluation_dock/high_affinity.yaml'
with open(config_dir, 'r') as f:
    config = yaml.full_load(f)

json_file = config['dock_dict_json']
dataset = config['dataset']

pocket_vina_path = config['pocket_vina_path']
final_result_path = config['final_result_path']


pocket_path = './data_crossdocked/test.yaml'           # './data_crossdocked/test.yaml'
ori_vina_path = '....csv'

with open(pocket_path, 'r') as f:
    pocket_dict = yaml.full_load(f)
pocket_names=list(pocket_dict.keys())




ori_vina = {}
with open(ori_vina_path, 'r') as file:
    csv_reader = csv.DictReader(file)
    for row in csv_reader:
        ligand_name = row['pocket_name']
        affinity = float(row['affinity'])
        ori_vina[ligand_name] = affinity




with open(json_file, 'r') as f:
    dock_data = json.load(f)

affinity_values = {}
for key, values in dock_data.items():
    for record in values:
        if record.get('mode_id') == 0:
            affinity_values[key] = record.get('affinity', None)
            break

affinity_list = [value for value in affinity_values.values() if value is not None]

print(affinity_list)

if affinity_list:
    average_affinity = sum(affinity_list) / len(affinity_list)
    print(f"Average Affinity: {average_affinity:.4f}")
else:
    print("No valid affinity values found.")


high_num =0
for key, affinity in affinity_values.items():
    pocket_name = re.sub(r'_\d$', '', key) + '.pdb'
    for key, value in ori_vina.items():
        if pocket_name in key:
            ori_affinity = value
            break
    if affinity <= ori_affinity:
        high_num = high_num + 1

print(high_num / len(affinity_list))


high_num=0
top_pocket={}
pocket_dock_values={}
for key,value in affinity_values.items():
    pocket_name = "_".join(key.split("_")[:-1])
    if pocket_name not in pocket_dock_values:
        pocket_dock_values[pocket_name]=[]
    pocket_dock_values[pocket_name].append((key,value))

all_molecules_affinities = {}
for pocket, molecules in pocket_dock_values.items():
    sorted_molecules = sorted(molecules, key=lambda x: x[1])[:40]
    for molecule, affinity in sorted_molecules:
        all_molecules_affinities[molecule] = affinity
print(sum(all_molecules_affinities.values()) / len(all_molecules_affinities))

for key, affinity in all_molecules_affinities.items():
    pocket_name = re.sub(r'_\d$', '', key) + '.pdb'
    for key, value in ori_vina.items():
        if pocket_name in key:
            ori_affinity = value
            break
    if affinity <= ori_affinity:
        high_num = high_num + 1

print(high_num / len(all_molecules_affinities))

# pocket_dock_values = defaultdict(list)
# for key, affinity in affinity_values.items():
#     pocket_key = re.sub(r'_\d+$', '', key) 
#     pocket_dock_values[pocket_key].append((key, affinity))

# all_molecules_affinities = {}
# for pocket, molecules in pocket_dock_values.items():
#     sorted_molecules = sorted(molecules, key=lambda x: x[1])[:10]
#     for molecule, affinity in molecules:
#         all_molecules_affinities[molecule] = affinity

# print(sum(all_molecules_affinities.values()) / len(all_molecules_affinities))

# high_num =0
# for molecule, affinity in all_molecules_affinities.items():
#     pocket_name = re.sub(r'_\d+$', '', molecule)
#     ori_ligand = pocket_name[:-9]
#     ori_ligand_vina = ori_vina[ori_ligand]
#     if affinity < ori_ligand_vina:
#         high_num = high_num + 1

# print(high_num)
# print(high_num / len(all_molecules_affinities))

# pocket_27 = []
# for key,value in pocket_dict.items():
#     pocket_item = os.path.splitext(os.path.basename(key))[0]
#     pocket_27.append(pocket_item)



# all_molecules_affinities_27 = {}
# for molecule, affinity in all_molecules_affinities.items():
#     pocket_name = re.sub(r'_\d+$', '', molecule)
#     if pocket_name in pocket_27:
#         all_molecules_affinities_27[molecule] = affinity

# high_num2=0
# for molecule, affinity in all_molecules_affinities_27.items():
#     pocket_name = re.sub(r'_\d+$', '', molecule)
#     ori_ligand = pocket_name[:-9]
#     ori_ligand_vina = ori_vina[ori_ligand]
#     if affinity < ori_ligand_vina:
#         high_num2 = high_num2 + 1

# print(sum(all_molecules_affinities_27.values()) / len(all_molecules_affinities_27))
# print(high_num2)
# print(high_num2 / len(all_molecules_affinities_27))

# pocket_min_values = {}
# high_num =0
# for key, affinity in affinity_values.items():
#     pocket_key = re.sub(r'_\d+$', '', key)

#     ori_ligand = pocket_key[:-9]
#     ori_ligand_vina = ori_vina[ori_ligand]

#     if affinity < ori_ligand_vina:
#         high_num = high_num + 1

#     if pocket_key not in pocket_min_values:
#         pocket_min_values[pocket_key] = affinity
#     else:
#         pocket_min_values[pocket_key] = min(pocket_min_values[pocket_key], affinity)

# high_num2 = 0
# for pocket_name, affinity in pocket_min_values.items():

#     ori_ligand = pocket_name[:-9]
#     ori_ligand_vina = ori_vina[ori_ligand]

#     if affinity < ori_ligand_vina:
#         high_num2 = high_num2 + 1
    

# print(len(pocket_min_values))

# print(sum(pocket_min_values.values()) / len(pocket_min_values))

# print(high_num / len(affinity_list))
# print(high_num2 / len(pocket_min_values))

