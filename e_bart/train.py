import os
import sys
from dataclasses import dataclass, field

import nltk
from datasets import load_dataset
from transformers import (
    HfArgumentParser,
    set_seed,
    AutoConfig,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    TrainingArguments
)

from transformers.trainer_utils import get_last_checkpoint

from model.modeling_e_bart import BartForConditionalGeneration
from ESeq2Seq_Trainer import ESeq2SeqTrainer

import numpy as np
import wandb
import json

import evaluate

import logging

logger = logging.getLogger(__name__)


@dataclass
class CustomArguments:
    # All arguments that are passed by the user to this program

    train_file: str = field(
        metadata={"help": "Provide a csv or json file for the training set"}
    )

    val_file: str = field(
        metadata={"help": "Provide a csv or json file for the validation set"}
    )

    test_file: str = field(
        metadata={"help": "Provide a csv or json file for the test set"}
    )

    train_guidance: str = field(
        metadata={"help": "Provide a csv or json file for the guidance signal"}
    )

    val_guidance: str = field(
        metadata={"help": "Provide a csv or json file for the guidance signal"}
    )

    test_guidance: str = field(
        metadata={"help": "Provide a csv or json file for the guidance signal"}
    )

    checkpoint_file: str = field(
        default=None, metadata={"help": "Provide a .bi file for resuming training or inference"}
    )

    max_train_samples: int = field(
        default=None, metadata={"help": "Provide a csv or json file for the training set"}
    )

    max_eval_samples: int = field(
        default=None, metadata={"help": "Provide a csv or json file for the training set"}
    )

    max_predict_samples: int = field(
        default=None, metadata={"help": "Provide a csv or json file for the training set"}
    )

    preprocessing_num_workers: int = field(
        default=None, metadata={"help": "The number of processes to use for the preprocessing."}
    )


def main():

    with open('model/config.json') as config_file:
        config_wandb = json.load(config_file)

    wandb.init(project="THESIS", config=config_wandb, name="EBART_V0")

    parser = HfArgumentParser((CustomArguments,Seq2SeqTrainingArguments))
    args, training_args = parser.parse_args_into_dataclasses()
    TrainingArguments.evaluation_strategy = "steps"
    TrainingArguments.eval_steps = 100

    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Do we train from scratch or do we  resume training from a checkpoint file ?
    # Detecting last checkpoint.

    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)

    # Set seed for reproducible behavior (set the seed in random, numpy and torch)
    set_seed(10)

    # Get the dataset
    data_files = {"train": args.train_file}
    data_files["validation"] = args.val_file
    data_files["test"] = args.test_file

    # Get the guidance
    guidance_files = {"train": args.train_guidance}
    guidance_files["validation"] = args.val_guidance
    guidance_files["test"] = args.test_guidance

    extension = args.test_file.split(".")[-1]

    raw_datasets = load_dataset(
        extension,
        data_files=data_files,
    )

    # Note that train, val and test guidance don't contain any  labels (x and no y) (only one column)
    # For now, we input as guidance the exact same as x but we will change this later

    raw_guidance = load_dataset(
        extension,
        data_files=guidance_files,
    )

    raw_datasets["train"] = raw_datasets["train"].add_column("guidance", raw_guidance.data["train"]["document"])
    raw_datasets["validation"] = raw_datasets["validation"].add_column("guidance", raw_guidance.data["validation"]["document"])
    raw_datasets["test"] = raw_datasets["test"].add_column("guidance", raw_guidance.data["test"]["document"])

    # Load configuration (sets architecture to follow, i.e. e_bart)
    my_config = AutoConfig.from_pretrained("model/config.json")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")

    # Load pretrained weights
    pretrained_weights = '/Users/clementgillet/Desktop/Master_Hub/ebart.large/model.bin'
    model = BartForConditionalGeneration.from_pretrained(pretrained_model_name_or_path=pretrained_weights, config=my_config)

    print(model)

    text_column = "document"
    summary_column = "summary"
    guidance_column = "guidance"

    max_source_length = 1024
    max_target_length = 240
    padding = "max_length"

    # The following is a nested funcion in main() that removes blanks in the dataset and tokenizes the input and the target
    # Return :

    def preprocess_function(examples):

        # remove pairs where at least one record is None

        inputs, guidance, targets = [], [], []
        for i in range(len(examples[text_column])):
            if examples[text_column][i] and examples[summary_column][i] and examples[guidance_column][i] :
                inputs.append(examples[text_column][i])
                targets.append(examples[summary_column][i])
                guidance.append(examples[guidance_column][i])

        inputs = [inp for inp in inputs]
        guidance = [guid for guid in guidance]

        # In NarraSum, the authors let truncation happen too.
        model_inputs = tokenizer(inputs, max_length=max_source_length, padding=padding, truncation=True)
        guidance = tokenizer(guidance, max_length=max_source_length, padding=padding, truncation=True)

        # Tokenize targets with the `text_target` keyword argument
        labels = tokenizer(text_target=targets, max_length=max_target_length, padding=padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length":
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        model_inputs["g"] = guidance["input_ids"]
        return model_inputs

    #TRAIN

    train_dataset = raw_datasets["train"]
    if args.max_train_samples is not None:
        max_train_samples = min(len(raw_datasets["train"]), args.max_train_samples)
        train_dataset = raw_datasets["train"].select(range(max_train_samples))

    """
    A context manager for torch distributed environment where on needs to do something on the main process, while blocking replicas, and when
    it's finished releasing the replicas.
    
    One such use is for `datasets`'s `map` feature which to be efficient should be run once on the main process, which upon completion saves 
    a cached version of results and which then automatically gets loaded by the replicas. 
    """

    with training_args.main_process_first(desc="train dataset map pre-processing"):
        train_dataset = train_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=train_dataset.column_names,
            desc="Running tokenizer on train dataset",
        )

    #VALIDATE
    eval_dataset = raw_datasets["validation"]
    if args.max_eval_samples is not None:
        max_eval_samples = min(len(raw_datasets["validation"]), args.max_eval_samples)
        eval_dataset = raw_datasets["validation"].select(range(max_eval_samples))

    with training_args.main_process_first(desc="validation dataset map pre-processing"):
        eval_dataset = eval_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=raw_datasets["validation"].column_names,
            desc="Running tokenizer on validation dataset",
        )


    #TEST
    predict_dataset = raw_datasets["test"]
    if args.max_predict_samples is not None:
        max_predict_samples = min(len(raw_datasets["test"]), args.max_predict_samples)
        predict_dataset = raw_datasets["test"].select(range(max_predict_samples))

    with training_args.main_process_first(desc="prediction dataset map pre-processing"):
        predict_dataset = predict_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=raw_datasets["test"].column_names,
            desc="Running tokenizer on prediction dataset",
        )

    # Data collator
    label_pad_token_id = -100
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
    )

    # Metric
    metric = evaluate.load("rouge")

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        # rougeLSum expects newline after each sentence
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        return preds, labels

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        # Replace -100s used for padding as we can't decode them
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        result = {k: round(v * 100, 4) for k, v in result.items()}
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        return result

    # Initialize our Trainer
    trainer = ESeq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset= eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics= compute_metrics,
    )

    # Training

    checkpoint = None
    if args.checkpoint_file is not None:
        checkpoint = args.checkpoint_file
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model()  # Saves the tokenizer too for easy upload

    metrics = train_result.metrics
    max_train_samples = (
        args.max_train_samples if args.max_train_samples is not None else len(train_dataset)
    )
    metrics["train_samples"] = min(max_train_samples, len(train_dataset))

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    # Evaluation
    results = {}
    logger.info("*** Evaluate ***")
    if isinstance(eval_dataset, dict):
        metrics = {}
        for eval_ds_name, eval_ds in eval_dataset.items():
            dataset_metrics = trainer.evaluate(eval_dataset=eval_ds, metric_key_prefix=f"eval_{eval_ds_name}")
            metrics.update(dataset_metrics)
    else:
        metrics = trainer.evaluate(metric_key_prefix="eval")
    max_eval_samples = args.max_eval_samples if args.max_eval_samples is not None else len(eval_dataset)
    metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)


    # Prediction
    logger.info("*** Predict ***")

    predict_results = trainer.predict(predict_dataset, metric_key_prefix="predict")
    metrics = predict_results.metrics
    max_predict_samples = (
        args.max_predict_samples if args.max_predict_samples is not None else len(predict_dataset)
    )
    metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

    trainer.log_metrics("predict", metrics)
    trainer.save_metrics("predict", metrics)

    if trainer.is_world_process_zero():
        if training_args.predict_with_generate:
            predictions = predict_results.predictions
            predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
            predictions = tokenizer.batch_decode(
                predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            predictions = [pred.strip() for pred in predictions]
            output_prediction_file = os.path.join(training_args.output_dir, "generated_predictions.txt")
            with open(output_prediction_file, "w") as writer:
                writer.write("\n".join(predictions))

    model_name_or_path = "EBART"
    dataset_name = "NarraSum"
    dataset_config_name = "Narrasum"
    lang  = "english"
    push_to_hub = False

    kwargs = {"finetuned_from": model_name_or_path, "tasks": "summarization"}
    if dataset_name is not None:
        kwargs["dataset_tags"] = dataset_name
        if dataset_config_name is not None:
            kwargs["dataset_args"] = dataset_config_name
            kwargs["dataset"] = f"{dataset_name} {dataset_config_name}"
        else:
            kwargs["dataset"] = dataset_name

    if lang is not None:
        kwargs["language"] = lang

    if push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)

    return results

if __name__ == '__main__':
    main()
