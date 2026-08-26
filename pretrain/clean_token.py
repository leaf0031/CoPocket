"""generate the vocabulary accorrding to the regular expressions of
SMILES of molecules."""
import yaml
import tqdm


def read_smi_to_list(file_path):
    smiles_list = []
    
    
    with open(file_path, 'r') as file:
        
        for line in file:

            clean_line = line.strip()

            smiles_list.append(clean_line)
    

    return smiles_list


def tokenize(smiles, tokens):
    """
    Takes a SMILES string and returns a list of tokens.
    Atoms with 2 characters are treated as one token. The 
    logic references this code piece:
    https://github.com/topazape/LSTM_Chem/blob/master/lstm_chem/utils/smiles_tokenizer2.py
    """
    n = len(smiles)
    tokenized = []
    i = 0

    # process all characters except the last one
    while (i < n - 1):
        # procoss tokens with length 2 first
        c2 = smiles[i:i + 2]
        if c2 in tokens:
            tokenized.append(c2)
            i += 2
            continue

        # tokens with length 2
        c1 = smiles[i]
        if c1 in tokens:
            tokenized.append(c1)
            i += 1
            continue

        raise ValueError(
            "Unrecognized charater in SMILES: {}, {}, {}".format(c1, c2,smiles))

    # process last character if there is any
    if i == n:
        pass
    elif i == n - 1 and smiles[i] in tokens:
        tokenized.append(smiles[i])
    else:
        raise ValueError(
            "Unrecognized charater in SMILES: {}".format(smiles[i]))
    return tokenized


if __name__ == "__main__":
    dataset_dir = "./datasets/chembl_clean_smi_len.smi"
    smiles_list = read_smi_to_list(dataset_dir)
    # output_smiles ='./datasets/chembl_smi_len_clean.smi'
    output_smiles ='./datasets/chembl_clean_smi_len_token2.smi'
    # output_smiles ='./datasets/chembl_smi_len_2.smi'

    atoms = [
        'Br', 'C', 'Cl', 'F', 'H', 'N', 'O', 'S'
    ]

    # atoms = [
    #     'Al', 'As', 'B', 'Br', 'C', 'Cl', 'F', 'H', 'I', 'K', 'Li', 'N',
    #     'Na', 'O', 'P', 'S', 'Se', 'Si', 'Te'
    # ]


    # special = [
    #     '(', ')', '[', ']', '=', '#', '%', '0', '1', '2', '3', '4', '5',
    #     '6', '7', '8', '9', '+', '-', 'se', 'te', 'c', 'n', 'o', 'p', 's'
    # ]

    special = [
        '(', ')', '[', ']', '=', '#', '%', '0', '1', '2', '3', '4', '5',
        '6', '7', '8', '9', '+', '-', 'c', 'n', 'o', 's'
    ]

    tokens = atoms + special
    tokens = set(tokens)

    print("computing token set from dataset...")


    with open(output_smiles, 'w') as file:
        for smiles in tqdm.tqdm(smiles_list, desc="Tokenizing SMILES"): 
            try:
                token = tokenize(smiles,tokens)
                if token:
                    file.write(smiles  + '\n')
            except:
                print("invalid:", smiles)
                continue
