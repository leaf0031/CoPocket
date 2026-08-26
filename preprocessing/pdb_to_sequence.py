from Bio.PDB import PDBParser, Polypeptide
from Bio import PDB
import json
import yaml
import os


def extract_sequence():

    pocket_path="/home/zql/Data/crossdock2020/crossdocked_pocket10/"
    # path="/home/zql/Code/HGNN-GPT/HGNN-RNN-Three/crossdocked_smiles/clean_test.yaml"
    path="/home/zql/Code/HGNN-GPT/HGNN-RNN-Four/data_crossdocked/raw/test.yaml"

    with open(path, 'r') as f:
        pocket_yaml = yaml.full_load(f)

    pockets = list(pocket_yaml.keys())

    pocket_sequence={}

    for item in pockets:
        pdb_file = item  

        pdb_path = os.path.join(pocket_path, pdb_file)

        try:
            full_sequence=extract_amino_acids_sequence(pdb_path)
            pocket_sequence[pdb_file]=full_sequence
        except Exception as e:
            print("wrong")

    # with open('/home/zql/Code/HGNN-GPT/HGNN-RNN-Four/data_crossdocked/raw/pocket_sequence_test.yaml', 'w') as file:
    with open('./case/test.yaml', 'w') as file:
        yaml.dump(pocket_sequence, file)


# # Extracting amino acid sequences from protein pockets
def extract_amino_acids_sequence(pocket_path):
    structure = PDB.PDBParser().get_structure("protein_structure", pocket_path)
    amino_acids = {}
    for model in structure:
        for chain in model:
            sequence = []
            for residue in chain:
                # Only check whether it is a standard amino acid residue.
                if Polypeptide.is_aa(residue):
                    if residue.id[0] == ' ':
                        try:
                            sequence.append(Polypeptide.three_to_one(residue.get_resname()))
                        except KeyError:
                            print(f"Unknown residue: {residue.get_resname()}")
            
            amino_acids[chain.id] = ''.join(sequence)  
    
    full_sequence = ''

    for chain, sequence in amino_acids.items():
        if sequence:  
            full_sequence += sequence

    return full_sequence

# extract_sequence()
pocket_path = '/home/disk2/xxr/now/HGNN-GPT-last-new-2__2/case1/7kzh.pdb'
sequence = extract_amino_acids_sequence(pocket_path)
print(sequence)
print(len(sequence))