# Why is Normalization Necessary for Linear Recommenders? (SIGIR'25)
This is the official code for DAN in the paper "[Why is Normalization Necessary for Linear Recommenders?](https://dl.acm.org/doi/10.1145/3726302.3730116)", [The 48th International ACM SIGIR Conference on Research and Development in Information Retrieval](https://sigir2025.dei.unipd.it/).

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/why-is-normalization-necessary-for-linear/collaborative-filtering-on-gowalla)](https://paperswithcode.com/sota/collaborative-filtering-on-gowalla?p=why-is-normalization-necessary-for-linear)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/why-is-normalization-necessary-for-linear/collaborative-filtering-on-yelp2018)](https://paperswithcode.com/sota/collaborative-filtering-on-yelp2018?p=why-is-normalization-necessary-for-linear)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/why-is-normalization-necessary-for-linear/collaborative-filtering-on-amazon-book)](https://paperswithcode.com/sota/collaborative-filtering-on-amazon-book?p=why-is-normalization-necessary-for-linear)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/why-is-normalization-necessary-for-linear/recommendation-systems-on-gowalla)](https://paperswithcode.com/sota/recommendation-systems-on-gowalla?p=why-is-normalization-necessary-for-linear)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/why-is-normalization-necessary-for-linear/recommendation-systems-on-yelp2018)](https://paperswithcode.com/sota/recommendation-systems-on-yelp2018?p=why-is-normalization-necessary-for-linear)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/why-is-normalization-necessary-for-linear/recommendation-systems-on-amazon-book)](https://paperswithcode.com/sota/recommendation-systems-on-amazon-book?p=why-is-normalization-necessary-for-linear)

## Introduction
Despite their simplicity, linear autoencoder (LAE)-based models have shown comparable or even better performance with faster inference speed than neural recommender models. However, LAEs face two critical challenges:

1. **Popularity bias**: LAEs tend to recommend popular items excessively
2. **Neighborhood bias**: LAEs overly focus on capturing local item correlations

To address these issues, we propose **Data-Adaptive Normalization (DAN)**, a versatile normalization solution that flexibly controls the popularity and neighborhood biases by adjusting item- and user-side normalization to align with unique dataset characteristics.

Note that a summary of our paper is on [our lab blog](https://dial.skku.edu/blog/2025_dan) (in Korean).

## Requirements
```
pip install -r requirements.txt
```

## How to run
The local adaptation uses the common TrainingFreeRecSys weak-generalization
splits rather than the upstream `valid_in.txt`/`test_in.txt` protocol:

```bash
TFRECSYS_DATA_ROOT=../../data/split_dataset/00_dataset_seed_2020 \
TFRECSYS_EVAL_FILE=test.txt \
python code/main.py --dataset gowalla --seed 2020 --model EASE_DAN \
  --reg_p 20 --alpha 0.2 --beta 0.3 --drop_p 0.6
```

Run through the common sweep framework:

```bash
cd ..
python run_sweep.py --models DAN --datasets gowalla --seeds 2020 \
  --verify_max_combinations 1
```

`EASE_DAN` is the sweep default. `LAE_DAN` and `RLAE_DAN` remain available
through `--model`. The bundled upstream datasets were removed. The loader now
reads `train.txt` and the file selected by `TFRECSYS_EVAL_FILE` from
`TFRECSYS_DATA_ROOT/<dataset>/`. Evaluation masks training interactions and
reports the same user-averaged Recall@20 and binary-relevance NDCG@20
definitions used by `EigendecompositionMF/code/model.py`.

`--solver auto` uses the upstream dense closed-form calculation on ordinary
catalogs and switches `EASE_DAN` to an algebraically equivalent Woodbury dual
solver when the item catalog is much larger than the user set. `--solver
primal` and `--solver dual` can be used to select either implementation
explicitly.

Adapted from upstream commit
`fc28a9542f141d13f495d5529aa3f2c486961356`.

## TODO
- [ ] Add the ML-20M, Netflix, and MSD datasets


## Citation

If you find our work useful for your research, please cite our paper:
```
@inproceedings{park2025dan,
  title={Why is Normalization Necessary for Linear Recommenders?},
  author={Seongmin Park and
          Mincheol Yoon and
          Hye-young Kim and
          Jongwuk Lee},
  booktitle={Proceedings of the 48th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  year={2025}
}
```
