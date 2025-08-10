# 任务三：Prompt任务（Prompt Tasks）-PET

## 一、实验概述

### 1.1 实验背景

PET (Pattern-Exploiting Training) 是一种基于prompt的少样本学习方法，其核心思想是：

1. 通过人工设计的自然语言模板将分类任务转化为完形填空任务

2. 利用预训练语言模型的MLM头进行预测

3. 通过verbalizer将预测词映射到真实标签

### 1.2 实验目标

复现PET方法在文本分类任务上的效果，验证基于模板的prompt学习在少样本场景下的有效性

## 二、数据集准备

使用项目中提供的用户评论预测用户评论的物品类别

## 三、模型训练

修改argparse所设置的参数，为节约时间，设置训练轮次为10

开启训练后，输出以下信息：

```
global step 5, epoch: 1, loss: 1.86494, speed: 0.09 step/s
Evaluation precision: 0.71000, recall: 0.61000, F1: 0.60000
best F1 performence has been updated: 0.00000 --> 0.60000
global step 10, epoch: 2, loss: 1.10287, speed: 0.07 step/s
Evaluation precision: 0.71000, recall: 0.69000, F1: 0.66000
best F1 performence has been updated: 0.60000 --> 0.66000
global step 15, epoch: 3, loss: 0.77830, speed: 0.07 step/s
Evaluation precision: 0.77000, recall: 0.72000, F1: 0.71000
best F1 performence has been updated: 0.66000 --> 0.71000
global step 20, epoch: 4, loss: 0.60103, speed: 0.09 step/s
Evaluation precision: 0.77000, recall: 0.77000, F1: 0.76000
best F1 performence has been updated: 0.71000 --> 0.76000
global step 25, epoch: 6, loss: 0.48565, speed: 0.07 step/s
Evaluation precision: 0.79000, recall: 0.77000, F1: 0.76000
global step 30, epoch: 7, loss: 0.40533, speed: 0.06 step/s
Evaluation precision: 0.79000, recall: 0.76000, F1: 0.74000
global step 35, epoch: 8, loss: 0.34781, speed: 0.07 step/s
Evaluation precision: 0.79000, recall: 0.74000, F1: 0.73000
global step 40, epoch: 9, loss: 0.30451, speed: 0.10 step/s
Evaluation precision: 0.79000, recall: 0.74000, F1: 0.73000
```

训练曲线图如下：

![Model Performance](D:\VSWorkSpace\Python\transformer\prompt_tasks\PET\output\Model%20Performance.png)

## 四、模型推理

完成模型训练后，运行 `inference.py` 以加载训练好的模型并应用：

```
...
contents = [
        '地理环境不错，但对面一直在盖楼，门前街道上打车不方便。',
        '跟好朋友一起凑单买的，很划算，洗发露是樱花香的，挺好的。。。'
    ]                           # 自定义评论
res = inference(contents)       # 推测评论类型
...
```

得到以下推理结果：

```
Prompt is -> 这是一条{MASK}评论：{textA}。
Used 0.26625990867614746s.
inference label(s):
['酒店', '洗浴']
```
