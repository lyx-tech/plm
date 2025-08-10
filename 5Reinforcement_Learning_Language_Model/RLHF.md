# 任务三：强化学习 & 语言模型

## 一、实验概述

随着 ChatGPT 的爆火，强化学习（Reinforcement Learning）和语言生成模型（Language Model）的结合开始变得越来越受人关注。

有关 ChatGPT 的视频讲解可以参考[这里](https://www.bilibili.com/video/BV1BG4y137SH/?vd_source=0df98e40ba56afac31703b0d5dba509f#reply143954452208)。

该项目的详细介绍可以参考[这里](https://mp.weixin.qq.com/s/1v4Uuc1YAZ9MRr1UWMH9xw)。

在这个项目中，我们将通过开源项目 [trl](https://github.com/lvwerra/trl) 搭建一个通过强化学习算法（PPO）来更新语言模型（GPT-2）的几个示例，包括：

* 基于中文情感识别模型的正向评论生成机器人（No Human Reward）

* 基于人工打分的正向评论生成机器人（With Human Reward）

* 基于排序序列（Rank List）训练一个奖励模型（Reward Model）

* 排序序列（Rank List）标注平台

## 二、模型训练

### 2.1 基于中文情感识别模型的正向评论生成机器人（No Human Reward）

修改config所设置的参数，为节约时间，设置训练轮次为2，步数为2000

开启训练后，输出以下信息：

```
  0%|          | 0/8 [00:00<?, ?it/s]epoch 0 mean-reward: 0.6669968366622925
Random Sample 5 text(s) of model output:
1. 这次购物总的来说体验很[SEP] [SEP] 很 满 意 包 装 很 好 是
2. 这部电影很俗 不 可 耐 的 展 现 出 来 了
3. 说实话，真的很般 般 ， 不 知 道 为 什 么 这
4. 刚收到货，感觉和 想 象 中 差 不 多 ， 比 想
5. 这次购物总的来说体验很悦 已 经 有 一 次 在 卓 越 购
 12%|����        | 1/8 [04:16<29:53, 256.18s/it]epoch 1 mean-reward: 0.6588645577430725
Random Sample 5 text(s) of model output:
1. 刚收到货，感觉用 了 一 周 ， 非 常 好 ， 非
2. 这次购物总的来说体验很[SEP] 老 板 是 个 非 常 有 气 质
3. 刚收到货，感觉用 着 还 不 错 ， 味 道 也 很
4. 这次购物总的来说体验很[SEP] 送 货 速 度 很 快 只 是 书
5. 这部电影很俗 ， 无 甚 艺 术 。 虽 然 每

 25%|������       | 2/8 [08:40<26:06, 261.02s/it]epoch 2 mean-reward: 0.8238215446472168
Random Sample 5 text(s) of model output:
1. 这部电影很诺 基 亚 的 不 错 [SEP] 诺 基 亚
2. 刚收到货，感觉诺 诺 的 很 漂 亮 ， 很 漂 亮
3. 刚收到货，感觉诺 诺 的 东 诺 诺 的 颜 色 很
4. 这部电影很诺 曼 底 登 陆 ， 对 于 我 们
5. 刚收到货，感觉诺 诺 的 质 量 应 该 不 会 差

 38%|��������      | 3/8 [13:07<21:58, 263.70s/it]epoch 3 mean-reward: 0.7855168581008911
Random Sample 5 text(s) of model output:
1. 这部电影很诺 贝 尔 文 章 的 风 格 很 好
2. 说实话，真的很诺 基 诺 的 手 机 真 的 很 不
3. 说实话，真的很诺 基 诺 基 诺 基 诺 基 诺 基
4. 刚收到货，感觉诺 基 诺 的 手 机 还 是 不 错
5. 这部电影很诺 贝 尔 文 章 的 风 格 很 好

 50%|����������     | 4/8 [17:30<17:34, 263.56s/it]epoch 4 mean-reward: 0.8605018258094788
Random Sample 5 text(s) of model output:
1. 这次购物总的来说体验很客 服 态 度 很 好 尤 其 是 听
2. 这部电影很客 的 风 格 ， 但 现 在 的 电
3. 说实话，真的很客 观 的 说 ， 这 是 一 家 很
4. 说实话，真的很客 家 菜 ， 尤 其 是 客 家 酿
5. 刚收到货，感觉客 服 态 度 很 好 ， 很 热 心

 62%|��������������   | 5/8 [21:47<13:03, 261.19s/it]epoch 5 mean-reward: 0.8643407821655273
Random Sample 5 text(s) of model output:
1. 刚收到货，感觉客 服 很 好 ， 很 热 情 ， 一
2. 这部电影很通 俗 易 懂 很 喜 欢 质 量 也
3. 这部电影很通 俗 易 懂 而 且 书 的 质 量
4. 刚收到货，感觉和 以 前 用 的 一 样 ， 从 外
5. 刚收到货，感觉， 比 较 不 错 ， 很 小 巧 ，  
 75%|����������������  | 6/8 [26:06<08:40, 260.35s/it]epoch 6 mean-reward: 0.8657691478729248
Random Sample 5 text(s) of model output:
1. 说实话，真的很， 我 们 买 了 一 个 特 价 的
2. 这部电影很通 俗 易 通 讲 述 了 一 个 人
3. 刚收到货，感觉， 好 大 一 个 ， 有 点 小 ，
4. 这部电影很， 很 好 吃 ， 我 很 喜 欢 ，
5. 刚收到货，感觉的 味 道 像 是 变 质 了 ， 还

 88%|������������������ | 7/8 [30:23<04:19, 259.32s/it]epoch 7 mean-reward: 0.8068673014640808
Random Sample 5 text(s) of model output:
1. 这次购物总的来说体验很， ， ， ， ， ， ， ， ， ，
2. 说实话，真的很， 估 计 质 量 不 好 ， 但 是
3. 说实话，真的很， 极 有 一 个 特 色 菜 ， 很
4. 刚收到货，感觉， 还 是 看 中 了 它 ， 特 别
5. 刚收到货，感觉的 菲 尼 尔 · 马 丁 ， 一 个

100%|��������������������| 8/8 [34:43<00:00, 259.57s/it]
100%|��������������������| 8/8 [34:43<00:00, 260.48s/it]
```

训练曲线图如下：

![PPO-Sentiment-Zh](D:\VSWorkSpace\Python\transformer\RLHF\output\PPO-Sentiment-Zh.png)

### 2.2 基于人工打分的评论生成机器人（With Human Reward）

启动标注平台，在终端看到模型的生成结果，通过人工输入 reward 以迭代模型：

```
[20:57:49] [Done] Initialized Model & Env.                                                          terminal_main.py:125
[Step 1]
prompt>>> 刚收到货，感觉
result>>> 形 状 挺 漂 亮 。 试 玩 了 一 下 ， 还 不 错 诶 。 进 去 试
Reward (-2.0 ~ 2.0): 1.3
[Step 2]
prompt>>> 这次购物总的来说体验很
result>>> 区 域 误 差 处 理 完 美 [SEP] 质 量 一 如 既 往 的 好 试 了 几
Reward (-2.0 ~ 2.0): -2.0
[Step 3]
prompt>>> 刚收到货，感觉
result>>> 吗 ？ 刚 收 到 货 ， 感 觉 还 没 有 到 过 五 星 级 的 速 度
Reward (-2.0 ~ 2.0): 1.5
[Step 4]
prompt>>> 说实话，真的很
result>>> 卫 生 很 差 服 务 也 也 很 差 什 么 都 比 较 老 半 当 眼 不
Reward (-2.0 ~ 2.0): 1.6
[Step 5]
prompt>>> 这次购物总的来说体验很
result>>> 层 次 不 错 [SEP] 型 很 好 买 来 中 低 端 用 [SEP] 合 身 。 建 议
Reward (-2.0 ~ 2.0): 1.9
[Step 6]
prompt>>> 刚收到货，感觉
result>>> 据 说 9 月 就 到 了 ， 阿 姨 手 脚 蛮 横 的 。 跟 他 说 了
Reward (-2.0 ~ 2.0): 1.9
[Step 7]
prompt>>> 刚收到货，感觉
result>>> 间 间 店 都 好 差 ， 老 公 说 广 州 南 大 街 街 门 口 呢 间
```

![Terminal-Human-Feedback](D:\VSWorkSpace\Python\transformer\RLHF\output2\Terminal-Human-Feedback.png)

### 2.3 基于人工排序训练 Reward Model

通过「排序序列」来学习一个「打分模型」。

训练数据集在 `data/reward_datasets/sentiment_analysis`，每一行是一个排序序列（用\t符号隔开）。排在越前面的越偏「正向情绪」，排在越后面越「负向情绪」。

```
1.买过很多箱这个苹果了，一如既往的好，汁多味甜～    2.名不副实。    3.拿过来居然屏幕有划痕，顿时就不开心了    4.什么手机啊！一台充电很慢，信号不好！退了！又买一台竟然是次品。
1.一直用沙宣的洗发露！是正品！去屑止痒润发护发面面俱到！    2.觉得比外买的稀，好似加了水的    3.非常非常不满意，垃圾。    4.什么垃圾衣服，买来一星期不到口袋全拖线，最差的一次购物
    ...
```

期望通过这个序列训练一个 Reward 模型，当句子越偏「正向情绪」时，模型给出的 Reward 越高。

开始训练后，设置训练次数为5，输出如下：

```
DatasetDict({
    train: Dataset({
        features: ['text'],
        num_rows: 12328
    })
    dev: Dataset({
        features: ['text'],
        num_rows: 3082
    })
})
global step 20, epoch: 1, loss: 0.37916, speed: 0.05 step/s
Evaluation acc: 0.44481
best F1 performence has been updated: 0.00000 --> 0.44481
global step 40, epoch: 1, loss: 0.34097, speed: 0.04 step/s
Evaluation acc: 0.45779
best F1 performence has been updated: 0.44481 --> 0.45779
```

由于设备限制，训练时间过长，还未能复现最后结果
