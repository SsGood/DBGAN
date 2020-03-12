# Distribution-induced Bidirectional GAN for Graph Representation Learning

This is a TensorFlow implementation of the Distribution-induced Bidirectional GAN (DBGAN) model as described in our paper.
Some of the code is borrowed from T. N. Kipf, M. Welling, Variational Graph Auto-Encoders [https://github.com/tkipf/gae]

## Introduction
This code contains two versions of the hyper-parameters. The first one is the implementation of node clustering task. The second one is the implementation of link prediction task.

## Requirements
* TensorFlow >= 1.12.0
* python 3.6
* networkx
* scikit-learn
* scipy
* dppy
* munkres

## Run from
preset version:
```bash
python run.py
```
or modifying the network parameters and run
```bash
python run.py --hidden3 xxx --hidden2 xxx --learning_rate xxx ...
```

You can select the dataset in ```run.py```

## Data

If you want to use your own data, you have to provide 
* an N by N adjacency matrix (N is the number of nodes), and
* an N by D feature matrix (D is the number of features per node) -- optional

Have a look at the `load_data()` function in `input_data.py` for an example.

In this example, we load citation network data (Cora, Citeseer or Pubmed). The original datasets can be found here: http://linqs.cs.umd.edu/projects/projects/lbc/ and here (in a different format): https://github.com/kimiyoung/planetoid
