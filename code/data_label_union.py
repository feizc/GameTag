#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 10:52:32 2020

@author: kaifeiwang
"""

"""脚本功能:提取三个搜索引擎的序列的交集并以pfind结果的格式存储"""
import csv

def get_peaks(Filename):
    """函数功能：提取peaks结果文件中的序列集合
    Filename:protein-peptides文件，.csv格式，第3列为序列内容"""
    res = set()
    with open(Filename, 'r') as f:
        lines = list(csv.reader(f))
        for i in range(1, len(lines)):
            res.add(lines[i][3].split('.')[1])
    print('peaks : ', len(res))
    return res
    

def get_MSFragger(Filename):
    """函数功能：提取MSFragger结果文件中的序列集合
    Filename:protein-peptides文件，第0列为序列内容"""
    res = set()
    with open(Filename, 'r') as f:
        f = f.readlines()
    for i in range(1, len(f)):
        res.add(f[i].split('\t', 1)[0])
    print('MSFragger : ', len(res))
    return res

def get_pFind(Filename):
    """函数功能：提取pFind结果文件中序列集合
    Filename:pFind-Filtered.spectra文件，第5列为序列内容"""
    res = set()
    with open(Filename, 'r') as f:
        f = f.readlines()
    for i in range(1, len(f)):
        res.add(f[i].split('\t', 6)[5])
    print('pFind : ', len(res))
    return res

def build(union, Filename):
    """函数功能：提取pFind结果文件中序列在交集中的行
    Filename:pFind-Filtered.spectra文件，第5列为序列内容
    union:三者交集集合"""
    res = ""
    with open(Filename, 'r') as f:
        f = f.readlines()
    res += f[0]  #存题头
    for i in range(1, len(f)):
        seq = f[i].split('\t', 6)[5]
        if seq in union:
            res += f[i]
    return res
    
if __name__ == "__main__":
    PEAKS_path = 'C:\\test_wkf\\Tag\\Peaks_result\\New Project 1\\New Project 1_PEAKS PTM_7\\protein-peptides.csv'
    pFind_path = 'C:\\test_wkf\\Tag\\mouse\\pfind_result\\pFindTask2\\result\\pFind-Filtered.spectra'
    MSFragger_path = 'C:\\test_wkf\\Tag\\mouse\\MSFragger_result\\peptide.tsv'
    new_path = 'C:\\test_wkf\\Tag\\mouse.txt'
   # temp = 'C:\\test_wkf\\Tag\\temp.txt'
    peaks = get_peaks(PEAKS_path)
    MSFragger = get_MSFragger(MSFragger_path)
    pfind = get_pFind(pFind_path)
    peaks_pfind = peaks.intersection(pfind)
    peaks_MSFragger = peaks.intersection(MSFragger)
    pfind_MSFragger = pfind.intersection(MSFragger)
    union = peaks_pfind.intersection(MSFragger)
    print('peaks_pfind ', len(peaks_pfind))
    print('peaks_MSFragger ', len(peaks_MSFragger))
    print('pfind_MSFragger ', len(pfind_MSFragger))
    print('union ', len(union))
    res = build(union, pFind_path)
    with open(new_path, 'w') as f:
        f.write(res)
        f.close()
    
    
    
    
    
