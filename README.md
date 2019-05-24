# Learning Deep Transformer Models for Machine Translation on Fairseq

The implementation of [Learning Deep Transformer Models for Machine Translation [ACL 2019] ](todo) (**Qiang Wang**, Bei Li, Tong Xiao, Jingbo Zhu, Changliang Li, Derek F. Wong, Lidia S. Chao)

> This code is based on [Fairseq v0.5.0](https://github.com/pytorch/fairseq/tree/v0.5.0)

## Installation

1. `pip install -r requirements.txt`
2. `python setup.py develop`
3. `python setup.py install`

## Prepare Training Data

1. Download the preprocessed [WMT'16 En-De dataset](https://drive.google.com/uc?export=download&id=0B_bZck-ksdkpM25jRUN2X2UxMm8) provided by Google to project root dir

2. Generate binary dataset at `data-bin/wmt16_en_de_google`

> `bash runs/prepare-wmt-en2de.sh`

## Train

### Train deep pre-norm baseline (20-layer encoder)

> `bash runs/train-wmt-en2de-deep-prenorm-baseline.sh`

### Train deep post-norm DLCL (25-layer encoder)

> `bash runs/train-wmt-en2de-deep-postnorm-dlcl.sh`

### Train deep pre-norm DLCL (30-layer encoder)

> `bash runs/train-wmt-en2de-deep-prenorm-dlcl.sh`

NOTE: BLEU will be calculated automatically when finishing training

## Results

Model | #Param. |Epoch* | BLEU 
:--|:--:|:--:|:--:|
[Transformer](https://arxiv.org/abs/1706.03762) (base) | 65M | 20 | 27.3
[Transparent Attention](https://arxiv.org/abs/1808.07561) (base, `16L`) | 137M | - | 28.0
[Transformer](https://arxiv.org/abs/1706.03762) (big) | 213M | 60 | 28.4
[RNMT+](https://arxiv.org/abs/1804.09849) (big) | 379M | 144 | 28.5
[Layer-wise Coordination](https://papers.nips.cc/paper/8019-layer-wise-coordination-between-encoder-and-decoder-for-neural-machine-translation.pdf) (big) | 210M* | - | 29.0
[Relative Position Representations](https://arxiv.org/abs/1803.02155) (big) | 210M | 60 | 29.2
[Deep Representation](https://arxiv.org/abs/1810.10181) (big) | 356M | - | 29.2
[Scailing NMT](https://arxiv.org/abs/1806.00187) (big) | 210M | 70 | 29.3
Our deep pre-norm Transformer (base, `20L`) | 106M | 20 | 28.9
Our deep post-norm DLCL (base, `25L`) | 121M | 20 | 29.2
Our deep pre-norm DLCL (base, `30L`) | 137M | 20 | 29.3


NOTE: `*` denotes approximate values.
