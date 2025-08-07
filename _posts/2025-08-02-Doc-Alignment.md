---
title: "端到端的文档定位与校准算法"
date: 2025-08-02 11:41:15 +/-0800
categories: [Programming]
tags: [Face, Computer Vision, CH]     # TAG names should always be lowercase
math: true
image: https://inspireface-1259028827.cos.ap-singapore.myqcloud.com/blogs_box/doc_loc.png
---

最近需要做一个类似扫描王那样的文档检测，搜索一下大多数网络上的资料都是类似使用传统算法去提取边缘信息再进行一些后处理得到文档轮廓。当然也有一些DNN的方案，比如hed边缘检测、语义分割纸张等。在当前年份也有人做一些比较深入的研究方向，比如弯曲的纸面使用非刚性变换去校准还原文档内容的方式。

当前我的需求比较简单，不需要考虑纸质弯曲变换，只考虑纸或证件放置于平面桌面的情况下即可。

最简单直观的方式就是找出文档的四个角点，然后使用透视变换进行对齐。

直接回归文档四个角的坐标点已经被证实了效果非常不好，原因大概是因为空间信息丢失，而且学习难度大，这边采用热力回归的方式，具体参考DSNT.

## DSNT原理简述

- 传统的 argmax 从热图提取坐标是**不可微**的：

$$\text{坐标} = \arg\max_{(x,y)} \text{热图}(x,y)$$

梯度无法反向传播，网络无法端到端训练。

- DSNT的解决方案是：

将热图转换为概率分布，然后计算**期望值**作为坐标。

### 步骤1：热图转概率（Spatial Softmax）

$$P(x,y) = \frac{e^{H(x,y)}}{\sum_{i,j} e^{H(i,j)}}$$

其中 $H$ 是网络输出的热图，$P$ 是归一化后的概率分布。

### 步骤2：计算期望坐标

$$\hat{x} = \sum_{x,y} P(x,y) \cdot x$$
$$\hat{y} = \sum_{x,y} P(x,y) \cdot y$$

这里的 $x, y$ 是预定义的坐标值（如归一化到 [-1, 1]）。

- 为什么DSNT可微：

期望值计算是**加权求和**，完全可微：

$$\frac{\partial \hat{x}}{\partial H} = \frac{\partial}{\partial H}\left[\sum P(x,y) \cdot x\right]$$

通过 softmax 的导数和链式法则，梯度可以传播回去：

$$\frac{\partial L}{\partial H} = P \cdot \left[(x - \hat{x})\frac{\partial L}{\partial \hat{x}} + (y - \hat{y})\frac{\partial L}{\partial \hat{y}}\right]$$