Positive and Unlabled Learning (pu-learning)
===========

A set of machine learning tools and algorithms for learning from positive and unlabled datasets.

Tools
-------

PUAdapter: A tool that adapts any estimator that can output a probability to positive-unlabled learning.
           It is based on: Elkan, Charles, and Keith Noto. "Learning classifiers from only positive and unlabeled data."
           Proceeding of the 14th ACM SIGKDD international conference on Knowledge discovery and data mining. 
           ACM, 2008.

## How to run scripts

### Running puAdapter Example
```bash
python -m src.examples.puAdapterExample
```

### Running Breast Cancer Example
```bash
python -m src.tests.breastCancer
```

