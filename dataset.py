import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
from atchley_features import compute_atchley_factors, get_atchley_table

def load_data(train_path, test_path, separator):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    train_df = pd.DataFrame({'seq1': train_df['seq_1'] + separator,
                             'seq2': train_df['seq_2'],
                             'label': train_df['label']})
    test_df = pd.DataFrame({'seq1': test_df['seq_1'] + separator,
                            'seq2': test_df['seq_2'],
                            'label': test_df['label']})
    
    dataset = DatasetDict({
        'train': Dataset.from_pandas(train_df),
        'test': Dataset.from_pandas(test_df)
    })
    
    return dataset

def tokenize_and_compute_atchley(dataset, tokenizer, max_length):
    seq1 = tokenizer(dataset['seq1'], padding='max_length', return_tensors='pt', max_length=max_length, truncation=True)
    seq2 = tokenizer(dataset['seq2'], padding='max_length', return_tensors='pt', max_length=max_length, truncation=True)

    atchley_factors_seq = compute_atchley_factors(dataset['seq1'], dataset['seq2'], get_atchley_table(), seq1_max_len=36, seq2_max_len=36)

    return {
        'input_ids': seq1['input_ids'],
        'input_ids2': seq2['input_ids'],
        'attention_mask': seq1['attention_mask'],
        'attention_mask2': seq2['attention_mask'],
        'atchley_factors_seq': atchley_factors_seq,
    }

def prepare_dataset(train_path, test_path, model_name, separator, max_length):
    dataset = load_data(train_path, test_path, separator)
    tokenizer = AutoTokenizer.from_pretrained(f'facebook/{model_name}')
    
    tokenized_dataset = dataset.map(
        lambda x: tokenize_and_compute_atchley(x, tokenizer, max_length),
        batched=True,
        batch_size=128
    )
    
    return tokenized_dataset, tokenizer