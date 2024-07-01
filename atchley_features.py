import pandas as pd
import numpy as np
import torch 
from torch.utils.data import Dataset


def get_atchley_table():
    data = "A,-0.591,-1.302,-0.733,1.570,-0.146;C,-1.343,0.465,-0.862,-1.020,-0.255;D,1.050,0.302,-3.656,-0.259,-3.242;E,1.357,-1.453,1.477,0.113,-0.837;F,-1.006,-0.590,1.891,-0.397,0.412;G,-0.384,1.652,1.330,1.045,2.064;H,0.336,-0.417,-1.673,-1.474,-0.078;I,-1.239,-0.547,2.131,0.393,0.816;K,1.831,-0.561,0.533,-0.277,1.648;L,-1.019,-0.987,-1.505,1.266,-0.912;M,-0.663,-1.524,2.219,-1.005,1.212;N,0.945,0.828,1.299,-0.169,0.933;P,0.189,2.081,-1.628,0.421,-1.392;Q,0.931,-0.179,-3.005,-0.503,-1.853;R,1.538,-0.055,1.502,0.440,2.897;S,-0.228,1.399,-4.760,0.670,-2.647;T,-0.032,0.326,2.213,0.908,1.313;V,-1.337,-0.279,-0.544,1.242,-1.262;W,-0.595,0.009,0.672,-2.128,-0.184;Y,0.260,0.830,3.097,-0.838,1.512"

    # Splitting the data
    split_data = [item.split(',') for item in data.split(';')]
    df = pd.DataFrame(split_data, columns=['amino.acid', 'f1', 'f2', 'f3', 'f4', 'f5'])

    # Converting columns to numeric
    df[['f1', 'f2', 'f3', 'f4', 'f5']] = df[['f1', 'f2', 'f3', 'f4', 'f5']].apply(pd.to_numeric)

    # Setting 'amino.acid' as index
    df.set_index('amino.acid', inplace=True)

    return (df)

def pad_sequence(sequence, desired_length, padding_value=[0, 0, 0, 0, 0]):
    current_length = len(sequence)
    if current_length < desired_length:
        padded_sequence = sequence + [padding_value] * (desired_length - current_length)
    else:
        padded_sequence = sequence[:desired_length]
    return padded_sequence

def encode_sequence_with_padding(sequence, atchley_df, desired_length):
    encoded_sequence = [atchley_df.loc[aa].values.tolist() if aa in atchley_df.index else [0, 0, 0, 0, 0] for aa in sequence]
    padded_encoded_sequence = pad_sequence(encoded_sequence, desired_length)
    return np.array(padded_encoded_sequence)

def calculate_atchley_map_with_padding(cdr3b_sequences, peptide_sequences, atchley_df, seq1_max_len, seq2_max_len):
    n_sequences = len(cdr3b_sequences)
    atchley_maps = np.zeros((n_sequences, 5, seq1_max_len, seq2_max_len), dtype=np.float16)
    for idx, (cdr3b_sequence, peptide_sequence) in enumerate(zip(cdr3b_sequences, peptide_sequences)):
        cdr3b_encoded = encode_sequence_with_padding(cdr3b_sequence, atchley_df, seq1_max_len)
        peptide_encoded = encode_sequence_with_padding(peptide_sequence, atchley_df, seq2_max_len)
        for i in range(seq1_max_len):
            for j in range(seq2_max_len):
                if not np.array_equal(cdr3b_encoded[i], [0,0,0,0,0]) and not np.array_equal(peptide_encoded[j], [0,0,0,0,0]):
                    atchley_maps[idx, :, i, j] = np.abs(cdr3b_encoded[i] - peptide_encoded[j])
    return atchley_maps

def compute_atchley_factors(sequence1,sequence2, atchley_df, seq1_max_len=36, seq2_max_len=36):
    # Assume this function is implemented elsewhere
    atchley_factors = calculate_atchley_map_with_padding(sequence1,sequence2, atchley_df, seq1_max_len, seq2_max_len)
    return torch.tensor(atchley_factors, dtype=torch.float)

