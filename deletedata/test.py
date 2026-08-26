from openbabel import pybel
import os
import subprocess
import yaml
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

def smi_pdb(smi,save_path):
    try:
        mol = pybel.readstring("smi", smi)
        # strip salt 
        mol.OBMol.StripSalts(10)
        mols = mol.OBMol.Separate()

        # print(pybel.Molecule(mols))

        mol = pybel.Molecule(mols[0])
        for imol in mols:
            imol = pybel.Molecule(imol)
            if len(imol.atoms) > len(mol.atoms):
                mol = imol

        # optimization
        mol.addh()
        mol.make3D(forcefield='mmff94', steps=100)
        mol.localopt()
        mol.write(format='pdb', filename=str(save_path), overwrite=True)
        return 1
    except:
        print(f"Tranformation of {smi} failed! ")
        return 0


def pdb_to_pdbqt(input_pdb, output_pdbqt):
    """使用 prepare_ligand4.py 将 PDB 转换为 PDBQT"""
    try:
        venv_python = ".../python"


        command = [
            venv_python,
            ".../prepare_ligand4.py",
            "-l", input_pdb,
            "-o", output_pdbqt
        ]


        subprocess.run(command, check=True)
        return True
    except:
        print(f"Tranformation of {input_pdb} failed! ")
        return False


def docking_with_sdf(protein_pdbqt, lig_pdbqt, centroid, verbose=1, out_lig_sdf=None, save_pdbqt=False):
    '''
    work_dir: is same as the prepare_target
    protein_pdbqt: .pdbqt file
    lig_sdf: ligand .sdf format file
    '''

    os.makedirs(save_pdbqt, exist_ok=True)
    os.makedirs(out_lig_sdf, exist_ok=True)

   
    cx, cy, cz = centroid

    out_lig_pdbqt = os.path.splitext(os.path.basename(lig_pdbqt))[0] + '_out.pdbqt'
    out_lig_pdbqt = os.path.join(save_pdbqt, out_lig_pdbqt)

    out_sdf_name = os.path.splitext(os.path.basename(lig_pdbqt))[0] + '_out.sdf'
    out_lig_sdf = os.path.join(out_lig_sdf, out_sdf_name)


    command = '''/home/zql/qvina/qvina2.1 \
        --receptor {receptor_pre} \
        --ligand {ligand_pre} \
        --center_x {centroid_x:.4f} \
        --center_y {centroid_y:.4f} \
        --center_z {centroid_z:.4f} \
        --size_x 50 --size_y 50 --size_z 50 \
        --out {out_lig_pdbqt} \
        --exhaustiveness {exhaust}
        obabel {out_lig_pdbqt} -O {out_lig_sdf} -h'''.format(receptor_pre = protein_pdbqt,
                                            ligand_pre = lig_pdbqt,
                                            centroid_x = cx,
                                            centroid_y = cy,
                                            centroid_z = cz,
                                            out_lig_pdbqt = out_lig_pdbqt,
                                            exhaust = 24,
                                            out_lig_sdf = out_lig_sdf)
    
    proc = subprocess.Popen(
            command, 
            shell=True, 
            stdin=subprocess.PIPE, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
        )
    proc.communicate()

    if not save_pdbqt:
        os.remove(out_lig_pdbqt)
    
    if verbose: 
        if os.path.exists(out_lig_sdf):
            print('searchable docking is finished successfully')
        else:
            print('docing failed')

    return out_lig_sdf


def calculate_center(pdbqt_file):
    parser = PDBParser()
    structure = parser.get_structure("pdbqt", pdbqt_file)

    coords = []
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    coords.append(atom.get_coord())
    coords = np.array(coords)
    center_of_mass = np.mean(coords, axis=0)
    center_of_mass = center_of_mass.astype(float)
    return center_of_mass

def get_result(docked_sdf, ref_mol=None):
    suppl = Chem.SDMolSupplier(docked_sdf,sanitize=False)
    results = []
    for i, mol in enumerate(suppl):
        if mol is None:
            continue
        line = mol.GetProp('REMARK').splitlines()[0].split()[2:]
        try:
            rmsd = CalcRMS(ref_mol, mol)
        except:
            rmsd = np.nan
        results.append(EasyDict({
            # 'rdmol': mol,
            'mode_id': i,
            'affinity': float(line[0]),
            'rmsd_lb': float(line[1]),
            'rmsd_ub': float(line[2]),
            # 'rmsd_ref': rmsd
        }))

    return results


def get_smi(config_dir):
    with open(config_dir, 'r') as f:
        config = yaml.full_load(f)
    return list(config.keys())


openbable_dir= './case_study/smi2pdbqt.yaml'
with open(openbable_dir, 'r') as f:
    config = yaml.full_load(f)


pocket_names = [config['pocket_name']]
smiles_yaml = config['smiles_yaml']
save_smiles_pdb = config['smiles_pdb']
save_smiles_pdbqt = config['smiles_pdbqt']
receptor_path = config['receptor_path']
out_path_sdf=config['out_path_sdf']
out_path_pdbqt=config['out_path_pdbqt']
save_prop_path=config['save_prop_path']

if not os.path.exists(save_smiles_pdb):
    os.makedirs(save_smiles_pdb)

if not os.path.exists(save_smiles_pdbqt):
    os.makedirs(save_smiles_pdbqt)

if not os.path.exists(out_path_sdf):
    os.makedirs(out_path_sdf)

if not os.path.exists(out_path_pdbqt):
    os.makedirs(out_path_pdbqt)




list_error=[]
error_2pdbqt = []
dock_dict={}

for index, pocket_item in enumerate(pocket_names):

    pocket_smiles_path = os.path.join(smiles_yaml, pocket_item)+ '_sampled_temp1.yaml'
    smiles = get_smi(pocket_smiles_path)

    centroid = calculate_center(receptor_path)

    for index, smile in enumerate(smiles):

        each_save_pdb = os.path.join(save_smiles_pdb, pocket_item) +'_' +str(index) + '.pdb'
        each_save_pdbqt = os.path.join(save_smiles_pdbqt, pocket_item) +'_' +str(index) + '.pdbqt'
        result = smi_pdb(smile, each_save_pdb)
        if result==0:
            error_item = pocket_item + '_' + str(index)
            list_error.append(error_item)
        else:
            result_pdbqt = pdb_to_pdbqt(each_save_pdb,each_save_pdbqt)
            if not result:
                error_item = pocket_item + '_' + str(index)
                error_2pdbqt.append(error_item)
        
        # each_save_pdbqt
        docked_sdf = docking_with_sdf(receptor_path,each_save_pdbqt,centroid,out_lig_sdf=out_path_sdf,save_pdbqt=out_path_pdbqt)
        result = get_result(docked_sdf)
        dock_dict['pocket'] = result


save_error_list = config['error_list']
with open(save_error_list +'error_2pdb.yaml', 'w') as f:
    yaml.dump(list_error, f)

with open(save_error_list +'error_2pdbqt.yaml', 'w') as f:
    yaml.dump(error_2pdbqt, f)

with open(save_error_list +'smi2pdbqt.yaml', 'w') as f:
    yaml.dump(config, f)

with open(save_prop_path + 'dock_dict.json', 'w') as f:
    json.dump(dock_dict, f, indent=4)
