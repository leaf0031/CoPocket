from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
import yaml 
import json
import os

# Convert molecular sdf to smiles
# Approach: Read train data from the .pt file, convert it to smiles, save the corresponding pocket index, and store it as train.yaml.
# Approach: Read test data from the .pt file, convert it to smiles, save the corresponding pocket index, and store it as test.yaml.

def sdf_to_smiles():
    pocket_path="/Data/crossdock2020/crossdocked_pocket10/"
    path="../data_crossdocked/output.json" #  data_crossdocked/raw/output.json


    with open(path, 'r') as f:
        data = json.load(f)

    pocket_smiles={}


    for item in data['test']:
        pdb_file = item[0]  
        sdf_file = item[1]  

        sdf_path = os.path.join(pocket_path, sdf_file)

        try:
            mol = Chem.MolFromMolFile(sdf_path)
            if mol is None:
                continue
            smi = Chem.MolToSmiles(mol)
            if smi is None or smi == '':
                continue
            print(smi)
            pocket_smiles[pdb_file]=smi
        except Exception as e:
            print("wrong")
        


    with open('sdf_smiles_test.yaml', 'w') as file:
        yaml.dump(pocket_smiles, file)

sdf_to_smiles()