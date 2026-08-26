from generate.sample_util import check_novelty, sample, canonic_smiles,inverted_dict,convert_smiles,obey_lipinski,calculate_diversity,set_seed
from rdkit.Chem import QED
from rdkit.Chem import Crippen
from rdkit import Chem
import pandas as pd
import numpy as np
import yaml

import metrics.SA_Score.sascorer as sascorer
import metrics.NP_Score.npscorer as npscorer
from dataloaders.dataloader import DatapairDataset
from torch_geometric.loader import DataLoader
from rdkit.Chem import Descriptors
from rdkit.Chem.rdMolDescriptors import CalcTPSA
from dataloaders.dataloader import pocket_sequence_gen
import warnings
warnings.filterwarnings("ignore")


# Note that this code needs to be placed in the same directory as train in order to run.

def safe_apply(func, molecule):
    try:
        return func(molecule)
    except Exception:
        return np.nan

test_dir='./filter_sequence_dssp_train.yaml'

with open(test_dir, "r") as file_c:
    data = yaml.safe_load(file_c)

molecule_data = [{"path": path, "smiles": smiles} for path, smiles in data.items()]

results = pd.DataFrame(molecule_data)

results["mol"] = results["smiles"].apply(Chem.MolFromSmiles)


# Batch calculation attributes
results['Ro5'] = results['mol'].apply(lambda x: obey_lipinski(x))
results['qed'] = results['mol'].apply(lambda x: safe_apply(QED.qed, x))
results['sas'] = results['mol'].apply(lambda x: safe_apply(sascorer.calculateScore, x))
results['sas0-1'] = results['mol'].apply(lambda x: safe_apply(
    lambda mol: round((10 - sascorer.calculateScore(mol)) / 9, 2), x
))
results['logp'] = results['mol'].apply(lambda x: safe_apply(Crippen.MolLogP, x))
results['tpsa'] = results['mol'].apply(lambda x: safe_apply(CalcTPSA, x))
results['np'] = results['mol'].apply(lambda x: safe_apply(npscorer.scoreMol, x))
results['weight'] = results['mol'].apply(lambda x: Descriptors.MolWt(x) )


filtered_results = results[(results['qed'] > 0.3) & (results['Ro5'] > 2)]


filtered_data = {row["path"]: row["smiles"] for _, row in filtered_results.iterrows()}


output_yaml_file = "final_filter_train.yaml"
with open(output_yaml_file, "w") as outfile:
    yaml.dump(filtered_data, outfile, default_flow_style=False)


print(f"Number of molecules screened:{len(filtered_results)}")

