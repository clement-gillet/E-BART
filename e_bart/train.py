import os
from dataclasses import dataclass, field

from datasets import load_dataset
from transformers import (
    HfArgumentParser,
    set_seed,
    AutoConfig,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
)

from transformers.trainer_utils import get_last_checkpoint

from model.modeling_e_bart import EBartModel
from ESeq2Seq_Trainer import ESeq2SeqTrainer


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
        default=None, metadata={"help": "If the training shall be resumed from a checkpoint file, the user inputs a .pt or .bin file"},
    )

    max_train_samples: int = field(
        default=None, metadata={"help": "Provide a csv or json file for the training set"}
    )

    max_val_samples: int = field(
        default=None, metadata={"help": "Provide a csv or json file for the training set"}
    )

    max_test_samples: int = field(
        default=None, metadata={"help": "Provide a csv or json file for the training set"}
    )

    preprocessing_num_workers: int = field(
        default=None, metadata={"help": "The number of processes to use for the preprocessing."}
    )


def main():
    parser = HfArgumentParser((CustomArguments,Seq2SeqTrainingArguments))
    args, training_args = parser.parse_args_into_dataclasses()

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
    raw_guidance = load_dataset(
        extension,
        data_files=guidance_files,
    )

    # Load configuration (sets architecture to follow, i.e. e_bart)
    my_config = AutoConfig.from_pretrained("model/config.json")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")

    # Load pretrained weights
    model = EBartModel(my_config)


    print(model)

    text_column = "document"
    summary_column = "summary"

    max_source_length = 1024
    max_target_length = 240
    padding = "max_length"

    # The following is a nested funcion in main() that removes blanks in the dataset and tokenizes the input and the target
    # Return :

    def preprocess_function(examples):

        # remove pairs where at least one record is None

        inputs, targets = [], []
        for i in range(len(examples[text_column])):
            if examples[text_column][i] and examples[summary_column][i]:
                inputs.append(examples[text_column][i])
                targets.append(examples[summary_column][i])

        inputs = [inp for inp in inputs]
        # In NarraSum, the authors let truncation happen too.
        model_inputs = tokenizer(inputs, max_length=max_source_length, padding=padding, truncation=True)

        # Tokenize targets with the `text_target` keyword argument
        labels = tokenizer(text_target=targets, max_length=max_target_length, padding=padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length":
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    train_dataset = raw_datasets["train"]
    train_guidance = raw_guidance["train"]
    if args.max_train_samples is not None:
        max_train_samples = min(len(raw_datasets["train"]), args.max_train_samples)
        train_dataset = raw_datasets["train"].select(range(max_train_samples))
        train_guidance = raw_guidance["train"].select(range(max_train_samples))

    """
    A context manager for torch distributed environment where on needs to do something on the main process, while blocking replicas, and when
    it's finished releasing the replicas.
    
    One such use is for `datasets`'s `map` feature which to be efficient should be run once on the main process, which upon completion saves 
    a cached version of results and which then automatically gets loaded by the replicas. 
    """

    # train_dataset.map() ...

    with training_args.main_process_first(desc="train dataset map pre-processing"):
        train_dataset = train_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=train_dataset.column_names,
            desc="Running tokenizer on train dataset",
        )
        train_guidance = train_guidance.map(
            preprocess_function,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=train_guidance.column_names,
            desc="Running tokenizer on train guidance",
        )

    # Data collator
    label_pad_token_id = -100
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
    )

    # Initialize our Trainer
    trainer = ESeq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        train_guidance=train_guidance,
        eval_dataset= None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics= None,
    )


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








if __name__ == '__main__':
    main()
