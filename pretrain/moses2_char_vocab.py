"""generate the vocabulary accorrding to the regular expressions of
SMILES of molecules."""
import yaml
from tqdm import tqdm
import pandas as pd


def read_smiles_file(path, column_name,percentage):
    data = pd.read_csv(path, engine='python')
    # smiles_list = data[column_name].tolist()
    smiles_list = data[column_name].apply(lambda x: x.strip() if isinstance(x, str) else x).tolist()
    num_data = len(smiles_list) 
    return smiles_list[0:int(num_data * percentage)]


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
        
        print(f"Error with SMILES: {smiles}")
        raise ValueError(
            "Unrecognized charater in SMILES: {}, {}".format(c1, c2))

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
    dataset_dir_moses = "../datasets/moses2.csv"            # ../datasets/guacamol2.csv
    output_vocab_moses = "./moses2_char_vocab.yaml"   # ./vocab/guacamol2_char_vocab.yaml

    dataset_dir_guacamol = "../datasets/guacamol2.csv"           
    output_vocab_guacamol = "./guacamol2_char_vocab.yaml"

    atoms = [
        'Al', 'As', 'B', 'Br', 'C', 'Cl', 'F', 'H', 'I', 'K', 'Li', 'N',
        'Na', 'O', 'P', 'S', 'Se', 'Si', 'Te'
    ]

    special = [
        '(', ')', '[', ']', '=', '#', '%', '0', '1', '2', '3', '4', '5',
        '6', '7', '8', '9', '+', '-', 'se', 'te', 'c', 'n', 'o', 'p', 's'
    ]

    tokens = atoms + special
    tokens = set(tokens)

    print("computing token set from dataset...")
    # smiles = read_smiles_file(dataset_dir_moses,'SMILES', 1)
    smiles = read_smiles_file(dataset_dir_guacamol,'smiles', 1)
    data_tokens = []
    [data_tokens.extend(tokenize(x, tokens)) for x in tqdm(smiles)]
    data_tokens = set(data_tokens)

    print("validating token set from dataset...")
    assert(data_tokens.issubset(tokens))
    print("OK")

    vocab_dict = {}
    for i, token in enumerate(tokens):
        vocab_dict[token] = i

    i += 1
    vocab_dict['<eos>'] = i
    i += 1
    vocab_dict['<sos>'] = i
    i += 1
    vocab_dict['<pad>'] = i

    with open(output_vocab_guacamol, 'w') as f:
        yaml.dump(vocab_dict, f)
