import sys
import os
from dataclasses import dataclass, field
from typing import Optional

import torch
import transformers

import wandb
from datasets import load_dataset
from transformers import HfArgumentParser, set_seed, AutoConfig


@dataclass
class Arguments:
    # All arguments that are passed by the user to this program

    checkpoint_file: str = field(
        default=None, metadata={
            "help": "If the training shall be resumed from a checkpoint file, the user inputs a .pt or .bin file"},
    )

    train_file: str = field(
        default=None, metadata={"help": "Provide a csv or json file for the training set"}
    )

    val_file: str = field(
        default=None, metadata={"help": "Provide a csv or json file for the validation set"}
    )

    test_file: str = field(
        default=None, metadata={"help": "Provide a csv or json file for the test set"}
    )


def main():
    print("In the making...")

    parser = HfArgumentParser(Arguments)
    args = parser.parse_args_into_dataclasses()[0]

    # Do we train from scratch or do we  resume training from a checkpoint file ?
    checkpoint = args.checkpoint_file

    # Set seed for reproducible behavior (set the seed in random, numpy and torch)
    set_seed(10)

    # Get the dataset
    data_files = {"train": args.train_file}
    extension = args.train_file.split(".")[-1]
    data_files["validation"] = args.val_file
    extension = args.val_file.split(".")[-1]
    data_files["test"] = args.test_file
    extension = args.test_file.split(".")[-1]

    raw_datasets = load_dataset(
        extension,
        data_files=data_files,
    )

    # Load pretrained model and tokenizer
    config = AutoConfig.from_pretrained(
        args.config_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )



if __name__ == '__main__':
    main()
