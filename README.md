# gokinjo

[![CircleCI](https://circleci.com/gh/momijiame/gokinjo.svg?style=svg)](https://circleci.com/gh/momijiame/gokinjo)

### What is this?

- A feature extraction library based on k-nearest neighbor algorithm in Python
  - k-NN based feature has experience of being used in 1st place solution of Kaggle competition (see references)
- Be able to switch backend of k-NN algorithm
  - [scikit-learn](https://github.com/scikit-learn/scikit-learn) (default)
  - [annoy](https://github.com/spotify/annoy)
- FYI: "gokinjo" is meant neighborhood in japanese.

### Prerequisite

- Python 3.6 or later
- setuptools >= 30.0.3.0

### How to install

#### From PyPI

```bash
$ pip install gokinjo
```

##### With annoy backend

```bash
$ pip install "gokinjo[annoy]"
```

#### From source code

```bash
$ pip install git+https://github.com/momijiame/gokinjo.git
```

### Usage example

- Please see [examples](https://github.com/momijiame/gokinjo/tree/master/examples) in GitHub repository.

### How to setup a development environment

```bash
$ pip install -e ".[develop]"
$ pytest
```

### References

- The competition which k-NN feature was used on 1st place solution
  - https://www.kaggle.com/c/otto-group-product-classification-challenge/discussion/14335
- R implementation
  - https://github.com/davpinto/fastknn
- Super respectable another Python implementation
  - https://github.com/upura/knnFeat
