# Introduction

This repository is an official implementation for **GameTag**, which can generate sequence tags for MS/MS with cooperative game learning algorithm.  

Please send any questions, comments or bug reports to feizhengcong@ict.ac.cn. 

GameTag is a sequence tag generation tool for MS/MS specra. GameTag accepts MS/MS spectra in the format of mgf, which can be transformed from other formats by [Open-pFind](http://pfind.ict.ac.cn/). 

# Requirements
* Python 3
* numpy
* [PyTorch](http://pytorch.org/) (>1.0)

# A Simple Test 

In this section, we provide a simple test to familiarize yourself with basic operations. 

1. Get the spectrum datasets you need to identify. Here, you can use a small dataset named human.mgf which can unzip from human.zip. This dataset is randomly selected from Mann-Human dataset (~ 5000+ spectra). You should put this dataset to file path: 
```
/data / human.mgf
```
 
2. Get the trained model provided by us in the file path: 
```
/data / CKPT
```
Actually, you don't need do anything, just take a look at this checkpoint. 

3. Use the model to deal with the dataset. You can just input the CMD as: 

```
python inference.py 
```
Yes! you can find the tool operating and the final identification results can be found in: 
```
/data / res.txt
```
The formation of res.txt is: 

20100826_Velos2_AnMi_SA_HeLa_4Da.364.364.3.0.dta   ->  spectrum id 
 
DHEVR    -> extracted tags

HQFWR 
 
HEVRR 
 
QFWRR


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
