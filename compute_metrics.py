import evaluate
import argparse
import json
import nltk
import torch
import logging
import numpy as np

from filelock import FileLock


logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(description="Compute metrics for a predictions file")
    parser.add_argument("--predictions_file", type=str, help="Path to the predictions JSONL file")
    parser.add_argument("--output_file", type=str, help="Path to the output file")
    parser.add_argument("--postprocess", action="store_true", default=False, help="Whether to postprocess the text")
    parser.add_argument("--inference_batch_size", type=int, default=128, help="Inference batch size for BLANC")
    parser.add_argument("--truncation", type=bool, help="Do you want to truncate the predictions to match the length of another prediction file ? ")
    parser.add_argument("--truncation_reference_file", type=str, help="Path to the predictions JSONL file to trim according to")
    return parser.parse_args()def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels

def main():
    args = parse_args()
    try:
        nltk.data.find("tokenizers/punkt")
    except (LookupError, OSError):
        with FileLock(".lock") as lock:
            nltk.download("punkt", quiet=True)
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except (LookupError, OSError):
        with FileLock(".lock") as lock:
            nltk.download("punkt_tab", quiet=True)

    logger.info(f"Reading predictions file from {args.predictions_file}...")
    predictions = []
    references = []
    documents = []

    if args.truncation:
        trim_lengths = []
        with open(args.truncation_reference_file, mode="r") as ref_file:
            for line in ref_file:
                truncation_reference = json.loads(line)
                trim_lengths.append(len(truncation_reference["generated_prediction"]))

        with open(args.predictions_file, mode="r") as f:
            for line, len in zip(f, trim_lengths):
                example = json.loads(line)
                predictions.append(example["generated_prediction"][:len])
                references.append(example["summary"])
                documents.append(example["document"])
    else:
        with open(args.predictions_file, mode="r") as f:
            for line in f:
                example = json.loads(line)
                predictions.append(example["generated_prediction"])
                references.append(example["summary"])
                documents.append(example["document"])

    logger.info("Computing metrics...")

    rouge = evaluate.load("rouge")
    if args.postprocess:
        # Relevant for rougeLSum
        rouge_predictions, rouge_references = postprocess_text(predictions, references)
        rouge_scores = rouge.compute(predictions=rouge_predictions, references=rouge_references, use_stemmer=True)
    else:
        rouge_scores = rouge.compute(predictions=predictions, references=references, use_stemmer=True)
    result_dict = {k: round(v * 100, 4) for k, v in rouge_scores.items()}
    logger.info(f"ROUGE scores:\n{result_dict}")

    meteor = evaluate.load("meteor")
    meteor_scores = meteor.compute(predictions=predictions, references=references)
    result_dict.update(meteor_scores)
    logger.info(f"METEOR score:\n{meteor_scores}")

    bert_score = evaluate.load("bertscore")
    bert_scores = bert_score.compute(predictions=predictions, references=references, lang="en")
    bert_scores["precision"] = np.mean(bert_scores.precision)
    bert_scores["recall"] = np.mean(bert_scores.recall)
    bert_scores["f1"] = np.mean(bert_scores.f1)
    result_dict["bert_score"] = bert_scores
    logger.info(f"BERTScore scores:\n{bert_scores}")

    blanc = evaluate.load("phucdev/blanc_score")
    blanc_scores = blanc.compute(
        summaries=predictions,
        documents=documents,
        blanc_score="help",
        device="cuda" if torch.cuda.is_available() else "cpu",
        inference_batch_size=128
    )
    blanc_score = np.mean(blanc_scores)
    result_dict.update(blanc_score)
    logger.info(f"BLANC score:\n{blanc_score}")

    with open(args.output_file, mode="w") as out_f:
        out_f.write(json.dumps(result_dict, indent=2))
    logger.info(f"Metrics saved to {args.output_file}")

if __name__ == "__main__":
    main()
