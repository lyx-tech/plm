# 任务六：文本生成（Text Generation）-Filling Model with T5

## 一、实验概述

### 1.1 实验背景

在信息抽取任务中，标注数据稀缺且成本高昂。传统数据增强方法（如同义词替换、回译）可能破坏文本的语义一致性。Mask Then Fill 策略通过生成模型（如 T5）对非关键信息片段进行掩码并重构，生成多样化且语义连贯的新样本，提升下游信息抽取模型的泛化能力。

### 1.2 实验目标

复现 Mask Then Fill 数据增强策略，基于 T5 模型实现非关键片段的掩码与生成。

## 二、数据集准备

项目中提供了一部分示例数据，数据来自DuIE数据集中的文本数据，为节约时间，本实验取0.01的数据进行抽样

## 三、模型训练

修改argparse所设置的参数，为节约时间，设置训练轮次为5

开启训练后，输出以下信息：

```
Sampled dataset length: 3501
global step 10, epoch: 1, loss: 9.65782, speed: 0.07 step/s
global step 20, epoch: 1, loss: 9.58952, speed: 0.08 step/s
Evaluation bleu4: 0.00000
best BLEU-4 performence has been updated: 0.00000 --> 0.00000
global step 30, epoch: 1, loss: 9.54636, speed: 0.05 step/s
global step 40, epoch: 1, loss: 9.48120, speed: 0.05 step/s
Evaluation bleu4: 0.00000
global step 50, epoch: 1, loss: 9.37707, speed: 0.05 step/s
global step 60, epoch: 2, loss: 9.26590, speed: 0.04 step/s
Evaluation bleu4: 0.00000
global step 70, epoch: 2, loss: 9.15878, speed: 0.05 step/s
global step 80, epoch: 2, loss: 9.05944, speed: 0.05 step/s
Evaluation bleu4: 0.00000
global step 90, epoch: 2, loss: 8.97155, speed: 0.05 step/s
global step 100, epoch: 2, loss: 8.89288, speed: 0.05 step/s
Evaluation bleu4: 0.00000
global step 110, epoch: 2, loss: 8.82480, speed: 0.05 step/s
global step 120, epoch: 3, loss: 8.76163, speed: 0.05 step/s
Evaluation bleu4: 0.00000
global step 130, epoch: 3, loss: 8.70572, speed: 0.05 step/s
global step 140, epoch: 3, loss: 8.65467, speed: 0.05 step/s
Evaluation bleu4: 0.00000
global step 150, epoch: 3, loss: 8.60843, speed: 0.05 step/s
global step 160, epoch: 3, loss: 8.56484, speed: 0.05 step/s
Evaluation bleu4: 0.00000
global step 170, epoch: 4, loss: 8.52569, speed: 0.05 step/s
global step 180, epoch: 4, loss: 8.48908, speed: 0.05 step/s
Evaluation bleu4: 0.00000
global step 190, epoch: 4, loss: 8.45585, speed: 0.05 step/s
global step 200, epoch: 4, loss: 8.42495, speed: 0.05 step/s
Evaluation bleu4: 0.00000
global step 210, epoch: 4, loss: 8.39573, speed: 0.05 step/s
global step 220, epoch: 4, loss: 8.36860, speed: 0.05 step/s
Evaluation bleu4: 0.00000
global step 230, epoch: 5, loss: 8.34359, speed: 0.05 step/s
global step 240, epoch: 5, loss: 8.32060, speed: 0.05 step/s
Evaluation bleu4: 0.00000
global step 250, epoch: 5, loss: 8.29831, speed: 0.05 step/s
global step 260, epoch: 5, loss: 8.27798, speed: 0.05 step/s
Evaluation bleu4: 0.00000
global step 270, epoch: 5, loss: 8.25873, speed: 0.05 step/s
```

训练曲线图如下：

![Model Performance](D:\VSWorkSpace\Python\transformer\plm\新建文件夹\Model%20Performance.png)

## 四、模型推理

完成模型训练后，运行 `inference.py` 以加载训练好的模型并应用：

```
 if __name__ == "__main__":
    masked_texts = [
        '"《μVision2单片机应用程序开发指南》是2005年2月[MASK]图书，作者是李宇"中[MASK]位置的文本是：'
    ]
    inference(masked_texts)
```

得到以下推理结果：由于抽样比例较小且训练轮次少，还未得到合理预测结果

```
maksed text: 
['"《μVision2单片机应用程序开发指南》是2005年2月[MASK]图书，作者是李宇"中[MASK]
位置的文本是：']
output: ['extra0']
```
