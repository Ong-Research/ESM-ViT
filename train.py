import yaml
import argparse
import torch
import numpy as np
import wandb
import random
from transformers import TrainingArguments, Trainer
import evaluate as evalhf
from model import EsmMMVit
from dataset import prepare_dataset

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def configure_trainer(config, tokenizer, tokenized_dataset):
    model = EsmMMVit.from_pretrained(f'facebook/{config["model_name"]}', num_labels=config['num_labels'])

    wandb.init(project=config['wandb_project'], name=f"{config['model_name']}_{config['separator']}_train")

    training_args = TrainingArguments(
        output_dir=f"{config['output_dir']}/{config['model_name']}_{config['separator']}",
        evaluation_strategy='steps',
        per_device_train_batch_size=config['per_device_train_batch_size'],
        per_device_eval_batch_size=config['per_device_eval_batch_size'],
        num_train_epochs=config['num_train_epochs'],
        logging_strategy='steps',
        learning_rate=config['learning_rate'],
        save_steps=config['save_steps'],
        save_total_limit=1,
        eval_steps=config['eval_steps'],
        load_best_model_at_end=True,
        metric_for_best_model="accuracy"
    )

    metric = evalhf.load('accuracy', experiment_id=random.randint(100,200))
    
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['test'],
        compute_metrics=compute_metrics
    )
    
    return trainer

def main(config_path):
    config = load_config(config_path)

    tokenized_dataset, tokenizer = prepare_dataset(
        config['train_path'], 
        config['test_path'], 
        config['model_name'], 
        config['separator'], 
        config['max_length']
    )

    trainer = configure_trainer(config, tokenizer, tokenized_dataset)

    train_result = trainer.train()
    wandb.finish()

    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    trainer.save_model()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ESM-ViT model")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    args = parser.parse_args()
    main(args.config)