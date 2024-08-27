# SFTFDA
Source-Free Time-Series Domain Adaptation (SFTSDA) approach addressing the absence of source-domain samples in the time-series unsupervised domain adaptation problems.

The code is based on the MAPU framework that taken from [github.com/mohamedr002/MAPU_SFDA_TS](https://github.com/mohamedr002/MAPU_SFDA_TS)

## Abstract
<img src="SFTSDA.png" width="500">
More research attention is still required for source-free time-series domain adaptations. However, current approaches rely on a simple pseudo-labelling strategy and require early memorization of noisy pseudo-labels. This study introduces Source-Free Time-Series Domain Adaptation as a new method to tackle the absence of source-domain samples in unsupervised time-series domain adaptation issues. SFTSDA utilizes a neighborhood pseudo-labelling approach that takes into account the predictions of a sample group when generating pseudo-labels. Additionally, it is trained to minimize contrastive loss so that samples of the same class are close to each other while those of different classes are far apart. An uncertainty reduction strategy is implemented to alleviate prediction uncertainties resulting from domain shifts. Lastly, curriculum learning strategy and the mean teacher framework are developed to address the issue of noisy pseudo-labels. Our experiments demonstrate the superiority of our approach over previous methods in benchmark problems with significant improvements.

## Requirmenets:
- Python3
- Pytorch==1.7
- Numpy==1.20.1
- scikit-learn==0.24.1
- Pandas==1.2.4
- skorch==0.10.0
- openpyxl==3.0.7 (for classification reports)
- Wandb=0.12.7 (for sweeps)

## Datasets
### Download Datasets
We used three public datasets in this study. We also provide the preprocessed versions as follows:
- [SSC](https://researchdata.ntu.edu.sg/dataset.xhtml?persistentId=doi:10.21979/N9/UD1IM9)
- [UCIHAR](https://researchdata.ntu.edu.sg/dataset.xhtml?persistentId=doi:10.21979/N9/0SYHTZ)
- [MFD](https://researchdata.ntu.edu.sg/dataset.xhtml?persistentId=doi:10.21979/N9/PU85XN)

### Training Procedure
The experiments are organised in a hierarchical way such that:
- Several experiments are collected under one directory assigned by --experiment_description
- Each experiment could have different trials, each is specified by --run_description.

## Train model
- run python trainers/train.py   
