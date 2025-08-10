# 任务二：信息抽取（Information Extraction）

## 一、实验概述

### 1.1 实验背景

UIE (Universal Information Extraction) 是一种统一的信息抽取框架，由百度PaddleNLP团队提出。它通过统一的Prompt机制，将多种信息抽取任务（如命名实体识别、关系抽取、事件抽取等）建模为相同的文本生成范式。

### 1.2 实验目标

复现UIE框架在中文信息抽取任务上的效果，验证其统一建模能力

## 二、数据集准备

项目中提供了一部分示例数据，数据来自DuIE数据集中随机抽取的100条

## 三、模型训练

修改argparse所设置的参数，为节约时间，设置训练轮次为2

开启训练后，输出以下信息：

```
Map:   0%|          | 0/232 [00:00<?, ? examples/s]
Map: 100%|��������������������| 232/232 [00:00<00:00, 3011.14 examples/s]
global step 20, epoch: 1, loss: 0.00402, speed: 0.11 step/s
Evaluation precision: 0.81633, recall: 0.88889, F1: 0.85106
best F1 performence has been updated: 0.00000 --> 0.85106
global step 40, epoch: 2, loss: 0.00274, speed: 0.18 step/s  
global step 60, epoch: 2, loss: 0.00225, speed: 0.09 step/s  
Evaluation precision: 0.75472, recall: 0.88889, F1: 0.81633
```

训练曲线图如下：

![Model Performance](D:\VSWorkSpace\Python\transformer\UIE\output\Model%20Performance.png)

## 四、模型推理

完成模型训练后，运行 `inference.py` 以加载训练好的模型并应用：

```
 if __name__ == "__main__":
    from rich import print
    sentences = [
        '谭孝曾是谭元寿的长子，也是谭派第六代传人。'
    ]

    # NER 示例
    for sentence in sentences:
        ner_example(
            model,
            tokenizer,
            device,
            sentence=sentence, 
            schema=['人物']
        )

    # SPO 抽取示例
    for sentence in sentences:
        information_extract_example(
            model,
            tokenizer,
            device,
            sentence=sentence, 
            schema={
                    '人物': ['父亲'],
                }
        )
```

NER和事件抽取在schema的定义上存在一些区别：

* NER的schema结构为 `List` 类型，列表中包含所有要提取的 `实体类型`。

* 信息抽取的schema结构为 `Dict` 类型，其中 `Key` 的值是所有 `主语`，`Value` 对应该主语对应的所有 `属性`。

* 事件抽取的schema结构为 `Dict` 类型，其中 `Key` 的值是所有 `事件触发词`，`Value` 对应每一个触发词下的所有 `事件属性`。

得到以下推理结果：

```

```
