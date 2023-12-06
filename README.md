# E-BART

Implementation of the Master Thesis [Leveraging Event Relation Extraction for Abstractive Summarization of Narratives: A Transformer-based Approach](https://drive.google.com/file/d/10BZcmVW58vcf13cZb0YOeEtXjysA3_Wk/view?usp=sharing), by Clément Gillet, 2023.

This repo contains all the code related to my **Master Thesis Research**. The **research question** is the following: 

"In the **narrative domain**, can we **improve performance of transformer-based summarizers**, i.e. quality and controllability of summaries, **by leveraging Event Relations** in the input document?"

## Architecture
<img src="images/gsum.png" width="40%" height="40%" alt="Architecture" title="Architecture">

#### Main Commands to operate:

- **Launch** =
  ```bash
python examples/pytorch/summarization/run_summarization.py \
    --model_name_or_path t5-small \
    --do_train \
    --do_eval \
    --dataset_name cnn_dailymail \
    --dataset_config "3.0.0" \
    --source_prefix "summarize: " \
    --output_dir /tmp/tst-summarization \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --overwrite_output_dir \
    --predict_with_generate
```
###
          train.py --train_file /ds/other/NarraSum/NarraSum/train.json --val_file /ds/other/NarraSum/NarraSum/validation.json --test_file /ds/other/NarraSum/NarraSum/test.json --train_guidance /ds/other/GS1_guid/train.json --val_guidance                  /ds/other/GS1_guid/validation.json --test_guidance /ds/other/GS1_guid/test.json --per_device_train_batch_size 4 --output_dir /netscratch/gillet/projects/E-BART/output/GS1 --max_eval_samples 300 --learning_rate 0.000003 --                        per_device_eval_batch_size 4 --max_target_length 250 --load_best_model_at_end True --num_train_epochs 3 --evaluation_strategy steps --predict_with_generate --metric_for_best_model rouge1

#### Python Requirements

Download [requirements.txt](requirements) and install all dependencies with following command : 

    pip install -r requirements.txt

## Datasets
- [NarraSum](https://github.com/zhaochaocs/narrasum)
- [MAVEN-ERE](https://github.com/THU-KEG/MAVEN-ERE)
- [MAVEN](https://github.com/THU-KEG/MAVEN-dataset)

## Installation

1.

