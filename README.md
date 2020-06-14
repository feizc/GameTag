# Introduction

This repository is an official implementation for **GameTag**, which can generate sequence tags for MS/MS with cooperative game learning.  

Please send any questions, comments or bug reports to feizhengcong@ict.ac.cn. 


GameTag is a sequence tag generation algorithm for MS/MS specra. GameTag accepts MS/MS spectra in the format of mgf, which can be  transformed from other formats by Open-pFind. 

# Requirements
* Python 3
* numpy
* [PyTorch](http://pytorch.org/) (>1.0)


# Training 

The basic training process is completed as: 

1. download the mass spectrum file (.mgf), labeled dataset and put it to file path: 
```
/data 
```
2. set the parameters such as where to store results file, model learning parameters, and hyper parameters. 

3. run the cmd: 
```
python train.py 
```
For the convenience of re-implementation, we provide the [mgf](https://pan.baidu.com/s/1yodL2z1cL7pqn_2Cnu1ydg) (password: j9pd) and [labeled dataset](https://pan.baidu.com/s/1t4vbJ_E2Pr1M4ajS93sQkg) (password: 7jbc) used in our experiments.


# Inference 

You can employ the model checkpoint provided by ours or the best performance model from training by yourself to generate the tag sequences. 

```
python inference.py 
```

# Evaluation 

We also provide an evaluation tool for tagging performance which can measure the metrics: sensitivity, tag coverage and average tag number automatically.  

```
python evaluate.py 
```
