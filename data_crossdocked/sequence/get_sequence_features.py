import yaml
import numpy as np
import os
import pickle
from tqdm import tqdm

from proteinbert import load_pretrained_model
from keras.models import Model

INPUT_FILE = 'data_crossdocked/sequence/pocket_sequence.yaml' 
OUTPUT_DIR = 'data/crossdocked_sequence_features/proteinbert' 
SEQ_LEN = 512  # Maximum sequence length

os.makedirs(OUTPUT_DIR, exist_ok=True)

pretrained_gen, input_encoder = load_pretrained_model()

full_model = pretrained_gen.create_model(seq_len=SEQ_LEN)

full_model.summary() 

target_layer_name = 'global-merge2-norm-block6'
target_output = full_model.get_layer(target_layer_name).output
feature_model = Model(inputs=full_model.input, outputs=target_output)

with open(INPUT_FILE, 'r') as f:
    protein_sequences = yaml.safe_load(f)

print(f"Begin extracting features from {len(protein_sequences)} protein sequences...")

for name, seq in tqdm(protein_sequences.items(), desc="extraction progress"):
    try:
        # coding sequence
        X, mask = input_encoder.encode_X([seq], SEQ_LEN)

        # Extract global embedding features (1, 512)
        global_repr = feature_model.predict([X, mask])
        features = global_repr[0]  
        print(features.shape)

        sub_dir = os.path.dirname(name)
        output_sub_dir = os.path.join(OUTPUT_DIR, sub_dir)
        os.makedirs(output_sub_dir, exist_ok=True)

        base_name = os.path.basename(name)
        file_name = os.path.splitext(base_name)[0] + '.pkl'
        feature_file = os.path.join(output_sub_dir, file_name)

        with open(feature_file, 'wb') as f:
            pickle.dump(features, f)

    except Exception as e:
        print(f"Error processing {name}: {e}")

print(f"\n Feature extraction complete! A total of {len(protein_sequences)} sequences were processed.")
print(f"Feature files are stored in: {OUTPUT_DIR}")
