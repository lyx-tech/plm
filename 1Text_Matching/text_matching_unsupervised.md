# 任务一：文本匹配（TextMatching）-无监督

## 一、实验概述

### 1.1 实验背景

文本匹配多用于计算两个文本之间的相似度，作为一种无监督模型，SimCSE使用droupout来对文本增加噪音，从而构造一个正样本对，而负样本对则是在batch中选取的其它句子。

### 1.2 实验目标

基于 SimCSE 实现一个无监督的文本匹配模型的训练流程。

## 二、数据集准备

使用项目中提供的未标注的用户搜索记录数据

## 三、模型训练

修改argparse所设置的参数，为节约时间，设置训练轮次为3

开启训练后，输出以下信息：

```
global step 20, epoch: 1, loss: 0.39844, speed: 0.11 step/s
Evaluation precision: 0.49412, recall: 1.00000, F1: 0.66142, spearman_corr: 
0.40841
best F1 performence has been updated: 0.00000 --> 0.66142
global step 40, epoch: 1, loss: 0.26816, speed: 0.11 step/s
Evaluation precision: 0.50000, recall: 1.00000, F1: 0.66667, spearman_corr: 
0.45767
best F1 performence has been updated: 0.66142 --> 0.66667
global step 60, epoch: 1, loss: 0.18874, speed: 0.10 step/s
Evaluation precision: 0.48837, recall: 1.00000, F1: 0.65625, spearman_corr: 
0.48454
global step 80, epoch: 2, loss: 0.14632, speed: 0.09 step/s
Evaluation precision: 0.49412, recall: 1.00000, F1: 0.66142, spearman_corr: 
0.43528
global step 100, epoch: 2, loss: 0.11994, speed: 0.09 step/s
Evaluation precision: 0.49412, recall: 1.00000, F1: 0.66142, spearman_corr: 
0.44155
global step 120, epoch: 2, loss: 0.10214, speed: 0.10 step/s
Evaluation precision: 0.49412, recall: 1.00000, F1: 0.66142, spearman_corr: 
0.45319
global step 140, epoch: 2, loss: 0.08896, speed: 0.10 step/s
Evaluation precision: 0.49412, recall: 1.00000, F1: 0.66142, spearman_corr: 
0.44961
```

训练曲线图如下：

![Model Performance](D:\VSWorkSpace\Python\transformer\text_matching\unsupervised\output\Model%20Performance.png)

## 四、模型推理

完成模型训练后，运行 `inference.py` 以加载训练好的模型并应用：

```
...
    if __name__ == '__main__':
    ...
    sentence_pair = [
        ('男孩喝女孩的故事', '怎样才知道是生男孩还是女孩'),
        ('这种图片是用什么软件制作的？', '这种图片制作是用什么软件呢？')
    ]
    ...
    res = inference(query_list, doc_list, model, tokenizer, device)
    print(res)
```

得到以下推理结果：

```
[0.37763088941574097, 0.9371953010559082]
```
