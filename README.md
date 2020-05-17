# Introduction
This repository is for **GameTag: A New Sequence Tag Generation Algorithm Based on Cooperative Game Theory**.

## Requirements
* Python 3
* numpy
* [PyTorch](http://pytorch.org/) (>1.0)


# Training 

The basic training process is completed as: 

1. download the mass spectrum file (.mgf), labeled dataset and put it to direction /data. 

2. set the parameters such as where to store results file, model learning parameters. 

3. run the cmd: python train.py 

For the convenience of demonstration, we provide the [mgf](https://pan.baidu.com/s/1yodL2z1cL7pqn_2Cnu1ydg) (password: j9pd) and [labeled dataset](https://pan.baidu.com/s/1t4vbJ_E2Pr1M4ajS93sQkg) (password: 7jbc) used in our experiments.


# Inference 

You can employ the model checkpoint provided by ours or the best performance model from training by yourself to generate the tag sequences. 


