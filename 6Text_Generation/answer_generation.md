# 任务六：文本生成（Text Generation）-Text-Generation, T5 Based

## 一、实验概述

### 1.1 实验背景

问答系统是自然语言处理中的重要应用，能够根据给定的问题和参考文本自动生成答案。本实验聚焦于生成式问答模型，使用T5（Text-to-Text Transfer Transformer）作为基础架构

T5模型将所有NLP任务统一为文本到文本的转换格式，这种统一框架使其特别适合生成式问答任务。

### 1.2 实验目标

复现基于T5的生成式问答模型，观察模型性能

## 二、数据集准备

实验使用百度开源的DuReaderQG数据集进行训练和评估，为节省时间，进行0.1的采样

## 三、模型训练

修改argparse所设置的参数，设置训练轮次为5

开启训练后，输出以下信息：

```
DatasetDict({
    train: Dataset({
        features: ['text'],
        num_rows: 14520
    })
    dev: Dataset({
        features: ['text'],
        num_rows: 984
    })
})

Map:   0%|          | 0/984 [00:00<?, ? examples/s]
Map: 100%|��������������������| 984/984 [00:00<00:00, 2097.49 examples/s]
Map: 100%|��������������������| 984/984 [00:00<00:00, 2086.57 examples/s]
Sampled dataset length: 1452
global step 10, epoch: 1, loss: 9.36240, speed: 0.06 step/s
global step 20, epoch: 1, loss: 9.22847, speed: 0.07 step/s
global step 30, epoch: 1, loss: 8.96581, speed: 0.07 step/s
Evaluation bleu4: 0.00000
best BLEU-4 performence has been updated: 0.00000 --> 0.00000
global step 40, epoch: 1, loss: 8.87838, speed: 0.04 step/s
global step 50, epoch: 2, loss: 8.77951, speed: 0.04 step/s
global step 60, epoch: 2, loss: 8.68294, speed: 0.04 step/s
Evaluation bleu4: 0.00000
best BLEU-4 performence has been updated: 0.00000 --> 0.00000
global step 70, epoch: 2, loss: 8.59820, speed: 0.04 step/s
global step 80, epoch: 2, loss: 8.51707, speed: 0.04 step/s
global step 90, epoch: 2, loss: 8.44394, speed: 0.04 step/s
Evaluation bleu4: 0.00000
global step 100, epoch: 3, loss: 8.37691, speed: 0.04 step/s
global step 110, epoch: 3, loss: 8.31401, speed: 0.04 step/s
global step 120, epoch: 3, loss: 8.25905, speed: 0.04 step/s
Evaluation bleu4: 0.00000
best BLEU-4 performence has been updated: 0.00000 --> 0.00000
global step 130, epoch: 3, loss: 8.20739, speed: 0.04 step/s
global step 140, epoch: 4, loss: 8.15759, speed: 0.04 step/s
global step 150, epoch: 4, loss: 8.11082, speed: 0.04 step/s
Evaluation bleu4: 0.00000
best BLEU-4 performence has been updated: 0.00000 --> 0.00000
global step 160, epoch: 4, loss: 8.06881, speed: 0.04 step/s
global step 170, epoch: 4, loss: 8.03035, speed: 0.04 step/s
global step 180, epoch: 4, loss: 7.99449, speed: 0.04 step/s
Evaluation bleu4: 0.00000
global step 190, epoch: 5, loss: 7.96171, speed: 0.04 step/s
global step 200, epoch: 5, loss: 7.93065, speed: 0.04 step/s
global step 210, epoch: 5, loss: 7.90130, speed: 0.04 step/s
Evaluation bleu4: 0.00000
global step 220, epoch: 5, loss: 7.87416, speed: 0.04 step/s
global step 230, epoch: 5, loss: 7.84963, speed: 0.04 step/s
```

训练曲线图如下：

![Model Performance](D:\VSWorkSpace\Python\transformer\answer_generation\output\Model%20Performance.png)

## 四、模型推理

完成模型训练后，运行 `inference.py` 以加载训练好的模型并应用：

```
...

if __name__ == '__main__':
    question = '治疗宫颈糜烂的最佳时间'
    context = '专家指出，宫颈糜烂治疗时间应选在月经干净后3-7日，因为治疗之后宫颈有一定的创面，如赶上月经期易发生感染。因此患者应在月经干净后3天尽快来医院治疗。同时应该注意，术前3天禁同房，有生殖道急性炎症者应治好后才可进行。'
    inference(qustion=question, context=context)
```



得到以下推理结果：

受采样和训练轮次影响，训练不够充分还未合理预测

```
Q: "治疗宫颈糜烂的最佳时间"
C: 
"专家指出，宫颈糜烂治疗时间应选在月经干净后3-7日，因为治疗之后宫颈有一定的创面
，如赶上月经期易发生感染。因此患者应在月经干净后3天尽快来医院治疗。同时应该注意
，术前3天禁同房，有生殖道急性炎症者应治好后才可进行。"
A: "extra0extra1extra2extra3extra4extra5extra6extra7extra20extra31"
```


