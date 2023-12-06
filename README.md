# E-BART

Implementation of the Master Thesis [Leveraging Event Relation Extraction for Abstractive Summarization of Narratives: A Transformer-based Approach](https://drive.google.com/file/d/10BZcmVW58vcf13cZb0YOeEtXjysA3_Wk/view?usp=sharing), by Clément Gillet, 2023.

This repo contains all the code related to my **Master Thesis Research**. The **research question** is the following: 

"In the **narrative domain**, can we **improve performance of transformer-based summarizers**, i.e. quality and controllability of summaries, **by leveraging Event Relations** in the input document?"

## Architecture
<img src="images/gsum.png" width="40%" height="40%" alt="Architecture" title="Architecture">

#### Main Commands to operate:

- **Launch** =

    python train.py --train_file /ds/other/NarraSum/NarraSum/train.json

#### Python Requirements

Download [requirements.txt](requirements) and install all dependencies with following command : 

    pip install -r requirements.txt

## Datasets
- [NarraSum](https://github.com/zhaochaocs/narrasum)
- [MAVEN-ERE](https://github.com/THU-KEG/MAVEN-ERE)
- [MAVEN](https://github.com/THU-KEG/MAVEN-dataset)

## Installation

1.

