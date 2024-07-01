import yaml
import argparse
import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, Trainer, TrainingArguments
from sklearn.metrics import classification_report, roc_auc_score
from model import EsmMMVit
from dataset import tokenize_and_compute_atchley
import os

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def configure_checkpoint(config, tokenizer, tokenized_dataset):
    model = EsmMMVit.from_pretrained(config['checkpoint_path'])

    training_args = TrainingArguments(
        output_dir=config['output_dir'],
        per_device_eval_batch_size=config['per_device_eval_batch_size'],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
    )
    
    return trainer

def main(config_path):
    config = load_config(config_path)
    print('Loading Trained model')

    # Load and prepare test data
    test_df = pd.read_csv(config['test_path'])
    X_test = test_df[['seq_1', 'seq_2']]
    y_test = test_df['label']

    test_df = pd.DataFrame({'seq1': X_test['seq_1'] + config['separator'], 'seq2': X_test['seq_2'],
                            'label': y_test})
    print('Test: ', test_df.label.value_counts())

    dataset = DatasetDict({
        'test': Dataset.from_pandas(test_df)
    })

    # Tokenize and prepare dataset
    tokenizer = AutoTokenizer.from_pretrained(f'facebook/{config["model_name"]}')
    tokenized_dataset = dataset.map(
        lambda x: tokenize_and_compute_atchley(x, tokenizer, config['max_length']),
        batched=True,
        batch_size=128
    )

    # Configure trainer with loaded checkpoint
    trainer = configure_checkpoint(config, tokenizer, tokenized_dataset)

    print('Running prediction')

    # Make predictions
    predictions, labels, metrics = trainer.predict(tokenized_dataset['test'], metric_key_prefix="predict")

    print(metrics)

    predy = np.argmax(predictions, axis=1)
    print(classification_report(y_test, predy))
    print('AUC Score: ', roc_auc_score(y_test, predictions[:, 1]))

    # get probabilities
    probabilities = sigmoid(predictions[:, 1])
    test_df['score'] = probabilities

    os.makedirs(args.output_dir, exist_ok=True)

    # Save results
    output_path = os.path.join(args.output_dir, 'test_results.csv')

    test_df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test ESM-ViT model")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    args = parser.parse_args()
    main(args.config)