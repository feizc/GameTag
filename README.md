# Introduction

This repository is an official implementation for **GameTag**, which can generate sequence tags for MS/MS with cooperative game learning algorithm.  

Please send any questions, comments or bug reports to feizhengcong@ict.ac.cn. 

GameTag is a sequence tag generation tool for MS/MS specra. GameTag accepts MS/MS spectra in the format of mgf, which can be transformed from other formats by [Open-pFind](http://pfind.ict.ac.cn/). 

# Requirements
* Python 3
* numpy
* [PyTorch](http://pytorch.org/) (>1.0)

# A Fast and Simple Test 

In this section, we provide a simple test to familiarize yourself with basic operations. 

1. Get the spectrum datasets you need to identify. Here, you can use a small dataset named human.mgf which can unzip from human.zip. This dataset is randomly selected from Mann-Human dataset (~ 5000+ spectra). You should put this dataset to file path: 
```
/data / human.mgf
```
 
2. Get the trained model provided by us in the file path: 
```
/data / CKPT
```
Actually, you don't need do anything, just take a check at this checkpoint. 

3. Use the GameTag model to deal with the dataset. You can just input the CMD as: 

```
python inference.py 
```
Yes! you can find the tool operating and the final identification results can be found in: 
```
/data / res.txt
```
The formation of res.txt is:  spectrum id  +  extracted tags. 



# Detailed Parameter Setting 

In this section, we will introduce the parameter setting in detail and customize the model you need in real application. 

## How to Evaluate Your own Dataset? 

1. You should prepare the dataset whose formation is cosistent with human.mgf provided by us. In fact, this is a very common form can be generated by, i.e., Open-pFind. 
 
2. A trained model is needed, you can employ the CKPT provided by us or trained by yourself (see next section). 

3. Utilize the trained model to deal with your dataset. For convenience,  two approaches are both available. 

a) CMD to determine the information: 

```
python inference.py --mgf_path [mgf] --model_path [model] --res_path [result] 
```

where --mgf_path denotes the path to the mgf file which we want to extract tag sequences, --model_path dentoes the path to trained model, and --res_path denotes the path to store the identification results. Take the simple test as example, we can write the command as: 

```
python inference.py --mgf_path /data/human.mgf --model_path /data/CKPT --res_path /data/res.txt 
```

b) change the python file: 

you can open the inference.py file, just change the parameter setting sections. 

for example, in the inference.py line 6-14:
```
# ------------------------------------------------------------------
# parameter setting
# path to the CKPT
model_path = './data/CKPT'
# path to the mgf which we want to extract tag
mgf_path = './data/human.mgf'
# path to store the results
res_path = './data/res.txt'
# ------------------------------------------------------------------
```
Just change the above variables and run the CMD: 
```
python inference.py 
```

Please note the above two approaches are completely equivalent, just select the one you like.  

Finally, we can find the results in the result file you specified.

## How to Train Your own Model?  

If you need to customize your own model, you can refer to this section. In short, you are supposed to prepare the annotated training dataset and fine tune the GameTag model. 
To be specific, we can: 

1. prepare the mgf dataset and label it with open-pfind. Of course, you can use other tools, such as PEAKS and MSFragger. 
At this point, you will hold two files, the mgf file and identification results from Open-pFind that serve as ground-truth. If you want to use the interaction of results from the three, just put the results file in the same file direction and run the cmd: 
```
python data_label_union.py
``` 
The formation of results file is consistent with Open-pFind, that is 1-th column and 5-th column are spectrum id and peptide sequence.

2. construct the training dataset according to the above two files. Run the cmd as: 
```
python label_dataset.py 
```
The training dataset is stored in the /data and named as labeled_tag.pkl by default.  


3. open the train.py python file, and modify the parameters such as where to store your trained model and tag length. Please note that if you want to change any parameters for any setting, just open the python file and you will find the parameter setting section in the top. Run the cmd as: 
```
python train.py 
``` 
After several iterations, we can find the best model in the /data directions. Multiple models may be saved and feel free to use anyone you like. 


For the convenience of re-implementation, we also provide the [mgf datasets](https://pan.baidu.com/s/1yodL2z1cL7pqn_2Cnu1ydg) (password: j9pd) and [labeled dataset](https://pan.baidu.com/s/1t4vbJ_E2Pr1M4ajS93sQkg) (password: 7jbc) used in our experiments.

