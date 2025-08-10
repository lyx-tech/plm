# 任务四：文本分类（Text Classification）

## 一、实验概述

### 1.1 实验背景

BERT-CLS 是基于 BERT 模型的文本分类方法，通过利用 BERT 的 [CLS] 标记的隐藏状态作为整个序列的表示，后接分类层进行文本分类。相比传统文本分类方法具有以下优势：

1. **上下文感知**：利用 Transformer 的自注意力机制捕捉全局上下文信息

2. **迁移学习**：通过预训练获得通用语言表示，在下游任务上微调即可获得良好效果

3. **多任务适应**：同一架构可应用于多种文本分类任务（情感分析、主题分类等）

### 1.2 实验目标

复现 BERT-CLS 在文本分类任务上的效果，验证其在多类别分类场景下的有效性

## 二、数据集准备

使用项目中提供的用户评论预测用户评论的物品类别

## 三、模型训练

修改argparse所设置的参数，为节约时间，设置训练轮次为5

开启训练后，输出以下信息：

```
global step 10, epoch: 1, loss: 2.03423, speed: 0.17 step/s
global step 20, epoch: 2, loss: 2.00665, speed: 0.16 step/s 
Evaluation precision: 0.07000, recall: 0.25000, F1: 0.11000
best F1 performence has been updated: 0.00000 --> 0.11000 
global step 30, epoch: 3, loss: 1.98826, speed: 0.12 step/s
global step 40, epoch: 4, loss: 1.98048, speed: 0.12 step/s 
Evaluation precision: 0.30000, recall: 0.25000, F1: 0.23000
best F1 performence has been updated: 0.11000 --> 0.23000 
global step 50, epoch: 4, loss: 1.96270, speed: 0.12 step/s
global step 60, epoch: 5, loss: 1.93882, speed: 0.12 step/s  
Evaluation precision: 0.26000, recall: 0.27000, F1: 0.25000
best F1 performence has been updated: 0.23000 --> 0.25000 
```

训练曲线图如下：

![Model Performance](D:\VSWorkSpace\Python\transformer\text_classification\output\Model%20Performance.png)

## 四、模型推理

完成模型训练后，运行 `inference.py` 以加载训练好的模型并应用：

```
...
sentences = [
        '外表跟图片不太像，而且号码偏大一些，穿着宽松而且裤腿不是很长，除了穿着暖和外还凑合吧，可能我个人试着不太合适',
        '不好看，不值这个价钱',
        '房间超级小，根本就不值688元的价格，特别是在广州这样一个酒店业十分发达的城市，酒店服务差，入住登记时强调要安静的房间。',
    ]
...
```

得到以下推理结果：

```
res: 
[7, 4, 7]
```
