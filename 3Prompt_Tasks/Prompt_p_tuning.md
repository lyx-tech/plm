# 任务三：Prompt任务（Prompt Tasks）-p-tuning

## 一、实验概述

### 1.1 实验背景

P-Tuning 是一种自动学习最优 prompt pattern 的方法，旨在解决传统离散 prompt 设计的不稳定性问题。传统的 prompt learning 依赖人工设计的模板，但不同 prompt 模板对模型性能影响很大，且人工设计成本高。P-Tuning 通过引入可学习的连续 prompt embeddings，让模型自动优化 prompt 模式，从而提升小样本学习能力

### 1.2 实验目标

复现 P-Tuning 在短文本分类任务（用户评论分类）上的效果，验证其在小样本场景下的优势。

## 二、数据集准备

使用项目中提供的用户评论预测用户评论的物品类别

## 三、模型训练

修改argparse所设置的参数，为节约时间，设置训练轮次为20

开启训练后，输出以下信息：

```
global step 5, epoch: 1, loss: 2.60481, speed: 0.32 step/s
global step 10, epoch: 2, loss: 1.51288, speed: 0.29 step/s
global step 15, epoch: 3, loss: 1.02161, speed: 0.31 step/s
global step 20, epoch: 4, loss: 0.76903, speed: 0.29 step/s
Evaluation precision: 0.72000, recall: 0.66000, F1: 0.64000
best F1 performence has been updated: 0.00000 --> 0.64000
global step 25, epoch: 6, loss: 0.61967, speed: 0.22 step/s
global step 30, epoch: 7, loss: 0.51715, speed: 0.22 step/s
global step 35, epoch: 8, loss: 0.44335, speed: 0.22 step/s
global step 40, epoch: 9, loss: 0.38808, speed: 0.22 step/s
Evaluation precision: 0.79000, recall: 0.78000, F1: 0.76000
best F1 performence has been updated: 0.64000 --> 0.76000
global step 45, epoch: 11, loss: 0.34501, speed: 0.22 step/s
global step 50, epoch: 12, loss: 0.31054, speed: 0.22 step/s
global step 55, epoch: 13, loss: 0.28233, speed: 0.22 step/s
global step 60, epoch: 14, loss: 0.25881, speed: 0.22 step/s
Evaluation precision: 0.77000, recall: 0.77000, F1: 0.75000
global step 65, epoch: 16, loss: 0.23892, speed: 0.22 step/s
global step 70, epoch: 17, loss: 0.22187, speed: 0.22 step/s
global step 75, epoch: 18, loss: 0.20710, speed: 0.22 step/s
global step 80, epoch: 19, loss: 0.19417, speed: 0.22 step/s
Evaluation precision: 0.76000, recall: 0.77000, F1: 0.75000
```

## 四、模型推理

完成模型训练后，运行 `inference.py` 以加载训练好的模型并应用：

```
...
contents = [
    "苹果卖相很好，而且很甜，很喜欢这个苹果，下次还会支持的", 
    "这破笔记本速度太慢了，卡的不要不要的"
]   # 自定义评论
res = inference(contents)       # 推测评论类型
...
```

得到以下推理结果：

```
Used 1.272674560546875s.
苹果卖相很好，而且很甜，很喜欢这个苹果，下次还会支持的 -> 水果
这破笔记本速度太慢了，卡的不要不要的 -> 平板
```
