"""
Pre-process the Chembl dataset:
    1. Normalize the molecule (this step also removes stereochemical info).
    2. Convert the SMILES to canonical form.
"""
from tqdm import tqdm
from rdkit import Chem, RDLogger
from rdkit.Chem import MolStandardize
from rdkit.Chem.MolStandardize import rdMolStandardize
import yaml
RDLogger.DisableLog('rdApp.*')


# Cleaning smiles, removing salt, and three-dimensionalization
class MolCleaner(object):
    def __init__(self):
        self.normarizer = MolStandardize.normalize.Normalizer()
        self.lfc = MolStandardize.fragment.LargestFragmentChooser()
        self.uc = MolStandardize.charge.Uncharger()

    def process(self, smi):
        mol = Chem.MolFromSmiles(smi)
        if mol:
            mol = self.normarizer.normalize(mol)
            mol = self.lfc.choose(mol)
            mol = self.uc.uncharge(mol)
            smi = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
            return smi
        else:
            return None



def smiles_cleaning(smiles):#Standardization
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        mol=rdMolStandardize.Normalize(mol)
        smi=Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
        return smi
    else:
        return None
    
def is_valid(smiles):#Ensure smiles are effective

    mol = Chem.MolFromSmiles(smiles)

    #If smiles is valid, return True; if invalid, return False.
    return smiles != '' and mol is not None and mol.GetNumAtoms() > 0


def read_dataset(dataset_path):
    with open(dataset_path, "r") as f:
        smiles_dict = yaml.full_load(f)
        # Dedupe, because the dictionary does not allow duplicate pocket names, key values cannot be the same.
    return smiles_dict

if __name__ == "__main__":
    in_path = "./sdf_smiles_train.yaml"
    out_path = "./clean_train.yaml"
    smiles_dict = read_dataset(in_path)
    smiles_list = list(smiles_dict.values())
    smiles_pocket= list(smiles_dict.keys())

    print("number of SMILES before cleaning:", len(smiles_list))

    # clean the molecules
    cleaner = MolCleaner()
    processed = []
    with open(out_path, "w") as f:
        for pocket,smiles in tqdm(smiles_dict.items()): 
            # mol=smiles_cleaning(smiles)
            smi = cleaner.process(smiles)
            if smi is not None:
    
                flag = is_valid(smi)

                if flag and smi is not None and 40 < len(smi) < 120:
                    processed.append(smi)
                    data={
                        pocket:smi     
                    }
                    documents = yaml.dump(data, f)

    print("number of SMILES after cleaning:", len(processed))

