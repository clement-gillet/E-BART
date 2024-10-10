# E-BART

Implementation of the Master Thesis [Leveraging Event Relation Extraction for Abstractive Summarization of Narratives: A Transformer-based Approach](https://drive.google.com/file/d/1siymb7ZUjV2eEBSF51tIDmV0mnFKAQHq/view?usp=sharing), by Clément Gillet, 2023.

This repo contains all the code related to my **Master Thesis Research**. The **research question** is the following: 

"In the **narrative domain**, can we **improve performance of transformer-based summarizers**, i.e. quality and controllability of summaries, **by leveraging Event Relations** in the input document?"

## Architecture
<img src="images/gsum.png" width="40%" height="40%" alt="Architecture" title="Architecture">


## Getting Started

This repository uses Python 3.9.
Download [requirements.txt](requirements.txt) and install all dependencies with following command : 

```bash
pip install -r requirements.txt
```

If you want to compute ROUGE, BERTScore, METEOR and BLANC with `compute_metrics.py`, you need to install the following packages:

```bash
pip install evaluate blanc==0.3.4 bert-score==0.3.13 rouge_score==0.1.2
```

## Preparation
In order to train E-BART, we need to provide event based guidance signals.
1. We do this by training a model for Event Detection on the MAVEN-ED dataset and predicting events for the NarraSum dataset: https://github.com/leonhardhennig/MAVEN-dataset/tree/debug/baselines/BERT%2BCRF/BERT-CRF-MAVEN
2. We prepare the augmented NarraSum dataset for Event Relation Extraction.
3. We then train a model for Event Relation Extraction on the MAVEN-ERE dataset and predict event relations for the event augmented NarraSum dataset: https://github.com/phucdev/MAVEN-ERE/tree/main/joint
4. We apply filtering to the augmented NarraSum dataset and convert the predicted events and event relations to guidance signals.


## Training and Evaluation
Keep in mind that you have to adjust all the paths to your own paths.
In order to track the training with `wandb` you need to log in with `wandb login` before running the scripts.

### Plain BART

```bash
python run_summarization.py \
    --model_name_or_path facebook/bart-large \
    --do_train \
    --do_eval \
    --do_predict \
    --train_file PATH/TO/YOUR/train.json \
    --validation_file PATH/TO/YOUR/valid.json \
    --test_file PATH/TO/YOUR/test.json \
    --output_dir PATH/TO/YOUR/E-BART/NarraSum_model/output/baseline \
    --per_device_train_batch_size=8 \
    --per_device_eval_batch_size=8 \
    --overwrite_output_dir \
    --predict_with_generate \
    --learning_rate 0.00003 \
    --text_column document \
    --summary_column summary \
    --run_name NarraSum_Baseline \
    --max_target_length 250 \
    --num_train_epochs 3 \
    --load_best_model_at_end True \
    --evaluation_strategy steps \
    --eval_steps 1000 \
    --save_steps 1000 \
    --metric_for_best_model rouge1 \
    --max_eval_samples 300
```

### GSum

```bash 
python e_bart/train.py \
    --train_file PATH/TO/YOUR/train.json \
    --val_file PATH/TO/YOUR/valid.json \
    --test_file PATH/TO/YOUR/test.json \
    --train_guidance PATH/TO/YOUR/train_guidance.json \
    --val_guidance PATH/TO/YOUR/valid_guidance.json \
    --test_guidance PATH/TO/YOUR/test_guidance.json \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --output_dir PATH/TO/YOUR/E-BART/NarraSum_model/output/e-bart \
    --max_eval_samples 300 \
    --pretrained_weights PATH/TO/YOUR/E-BART/model.bin \
    --learning_rate 0.000003 \
    --per_device_eval_batch_size 4 \
    --max_target_length 250 \
    --run_name NarraSum_E-BART \
    --load_best_model_at_end True \
    --num_train_epochs 3 \
    --evaluation_strategy steps \
    --predict_with_generate \
    --metric_for_best_model rouge1
```

## Compute additional metrics

The training scripts will compute ROUGE scores during the evaluation and export the predicted summaries along with the 
original documents and reference summaries in a file called `generated_predictions.jsonl`.
In order to compute additional metrics (BERTScore, METEOR, BLANC), you can use the `compute_metrics.py` script:

```bash
python compute_metrics.py \
    --predictions_file PATH/TO/YOUR/E-BART/NarraSum_model/output/e-bart/generated_predictions.jsonl \
    --output_file PATH/TO/YOUR/E-BART/NarraSum_model/output/e-bart/predictions_metrics.json \
    --postprocess
```
The `--postprocess` will do some processing for ROUGE evaluation because rougeLSum expects newline after each sentence.

## Datasets
- [NarraSum](https://github.com/zhaochaocs/narrasum)
- [MAVEN-ERE](https://github.com/THU-KEG/MAVEN-ERE)
- [MAVEN](https://github.com/THU-KEG/MAVEN-dataset)
