# 任务一：文本匹配（TextMatching）-监督

## 一、实验概述

### 1.1 实验背景

文本匹配是自然语言处理（NLP）领域中的一项重要任务，它可以用于很多应用，比如信息检索、机器翻译、对话系统等。文本匹配的目的是判断两个文本是否具有一定的相似度或者关系，通常通过计算它们之间的相似度来进行判断。

### 1.2 实验目标

对3种常用的文本匹配的方法进行实现：PointWise（单塔）、DSSM（双塔）、Sentence BERT（双塔）。

## 二、数据集准备

使用项目中提供的示例数据，利用「商品评论」和「商品类别」来进行文本匹配任务

## 三、模型训练

### 3.1 PointWise（单塔）

#### 3.1.1 模型训练

修改argparse所设置的参数，为节约时间，设置训练轮次为3

开启训练后，输出以下信息：

```
global step 10, epoch: 1, loss: 0.53384, speed: 0.06 step/s
Evaluation precision: 0.86275, recall: 0.83019, F1: 0.84615
best F1 performence has been updated: 0.00000 --> 0.84615
global step 20, epoch: 2, loss: 0.38735, speed: 0.05 step/s
Evaluation precision: 0.90291, recall: 0.87736, F1: 0.88995
best F1 performence has been updated: 0.84615 --> 0.88995
global step 30, epoch: 2, loss: 0.30701, speed: 0.05 step/s
Evaluation precision: 0.80328, recall: 0.92453, F1: 0.85965
global step 40, epoch: 3, loss: 0.26204, speed: 0.04 step/s
Evaluation precision: 0.93069, recall: 0.88679, F1: 0.90821
best F1 performence has been updated: 0.88995 --> 0.90821
```

可以看到模型很快达到较高精度，体现单塔模型准确率高的优势

训练曲线图如下：

![Pointwise Model Performance](D:\VSWorkSpace\Python\transformer\plm\1Text_Matching\Pointwise%20Model%20Performance.png)

#### 3.1.2 模型推理

模型训练后，运行`inference_pointwise.py` 以加载训练好的模型并应用：

```
test_inference(
        '手机：一种可以在较广范围内使用的便携式电话终端。',
        '味道非常好，京东送货速度也非常快，特别满意。',
        max_seq_len=128
    )
```

得到以下推理结果：

```
tensor([[ 1.8559, -2.1140]])
```

### 3.2 DSSM（双塔）

#### 3.2.1 模型训练

修改argparse所设置的参数，为节约时间，设置训练轮次为3

开启训练后，输出以下信息：

```
global step 0, epoch: 1, loss: 0.40534, speed: 0.23 step/s
Evaluation precision: 0.30114, recall: 1.00000, F1: 0.46288
best F1 performence has been updated: 0.00000 --> 0.46288
global step 10, epoch: 1, loss: 0.28879, speed: 0.02 step/s
Evaluation precision: 0.43192, recall: 0.86792, F1: 0.57680
best F1 performence has been updated: 0.46288 --> 0.57680
global step 20, epoch: 2, loss: 0.23583, speed: 0.02 step/s
Evaluation precision: 0.59281, recall: 0.93396, F1: 0.72527
best F1 performence has been updated: 0.57680 --> 0.72527
global step 30, epoch: 3, loss: 0.20403, speed: 0.02 step/s
Evaluation precision: 0.62000, recall: 0.87736, F1: 0.72656
best F1 performence has been updated: 0.72527 --> 0.72656
global step 40, epoch: 3, loss: 0.18068, speed: 0.02 step/s
Evaluation precision: 0.64103, recall: 0.94340, F1: 0.76336
best F1 performence has been updated: 0.72656 --> 0.76336
```

双塔模型的准确度低于单塔模型

训练曲线图如下：

![DSSM Model Performance](D:\VSWorkSpace\Python\transformer\plm\1Text_Matching\DSSM%20Model%20Performance.png)

#### 3.2.2 模型推理

和单塔模型不一样的是，双塔模型可以事先计算所有候选类别的Embedding，当新来一个句子时，只需计算新句子的Embedding，并通过余弦相似度找到最优解即可。

因此，在推理之前，我们需要提前计算所有类别的Embedding并保存。

保存的embedding文件如下：

```
{"0": {"label": "水果", "text": "水果：指多汁且主要味觉为甜味和酸味，可食用的植物果实。", "embedding": [0.9998800158500671, 0.9940347671508789, 0.9999966025352478, 0.9944102764129639, 0.9638229012489319, 0.9798339009284973, -0.9868173003196716, -0.6569762825965881, -0.9844879508018494, -0.9999186396598816, ...
```

模型推理结果如下：

输出（类别，余弦相似度）的二元组，并按照相似度做倒排（相似度取值范围：[-1, 1]）。

```
Used 0.18916797637939453s.
[
    ('平板', 0.8501089215278625),
    ('电脑', 0.4973188638687134),
    ('手机', 0.4318808317184448),
    ('酒店', 0.3925189673900604),
    ('衣服', 0.37076523900032043),
    ('洗浴', 0.31228092312812805),
    ('书籍', 0.14662379026412964),
    ('水果', 0.12923386693000793),
    ('电器', -0.019846543669700623),
    ('蒙牛', -0.06483691185712814)
]
```

### 3.3 Sentence Transformer（双塔）

#### 3.3.1 模型训练

修改argparse所设置的参数，为节约时间，设置训练轮次为3

开启训练后，输出以下信息：

```
global step 0, epoch: 1, loss: 0.72877, speed: 0.34 step/s
Evaluation precision: 0.29195, recall: 0.82075, F1: 0.43069
best F1 performence has been updated: 0.00000 --> 0.43069
global step 10, epoch: 1, loss: 0.66837, speed: 0.02 step/s
Evaluation precision: 0.44974, recall: 0.80189, F1: 0.57627
best F1 performence has been updated: 0.43069 --> 0.57627
global step 20, epoch: 2, loss: 0.61910, speed: 0.02 step/s
Evaluation precision: 0.54348, recall: 0.23585, F1: 0.32895
global step 30, epoch: 3, loss: 0.58015, speed: 0.02 step/s
Evaluation precision: 0.65741, recall: 0.66981, F1: 0.66355
best F1 performence has been updated: 0.57627 --> 0.66355
global step 40, epoch: 3, loss: 0.54516, speed: 0.02 step/s
Evaluation precision: 0.72642, recall: 0.72642, F1: 0.72642
best F1 performence has been updated: 0.66355 --> 0.72642

```

训练曲线图如下：

![Sentence Transformer Model Performance](D:\VSWorkSpace\Python\transformer\plm\1Text_Matching\Sentence%20Transformer%20Model%20Performance.png)

#### 3.3.2 模型推理

作为双塔模型，先计算所有候选文本的embedding值。

保存的embedding文件如下：

```
{"0": {"label": "水果", "text": "水果：指多汁且主要味觉为甜味和酸味，可食用的植物果实。", "embedding": [1.0910050868988037, -0.2556517422199249, -0.7220128774642944, -0.33099186420440674, 0.6600499153137207, -0.28853246569633484, 0.23967792093753815, 0.8968705534934998, 
```

模型推理结果如下：

函数将输出（匹配通过的类别，匹配值）的二元组，并按照匹配值（越大则越匹配）做倒排。

```
Used 0.16383123397827148s.
[
    ('电脑', 0.1260472983121872),
    ('平板', -0.08108583092689514),
    ('手机', -0.13328678905963898)
]
```
