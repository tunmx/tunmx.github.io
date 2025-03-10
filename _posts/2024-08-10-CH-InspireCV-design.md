---
title: "自定义轻量化计算机视觉库"
date: 2024-08-10 04:35:41 +/-0800
categories: [Programming]
tags: [Face, Computer Vision, CH]     # TAG names should always be lowercase
math: true
image: https://tunm-resource.oss-cn-hongkong.aliyuncs.com/blogs_box/u%3D2454171237%2C1300750673%26fm%3D253%26fmt%3Dauto%26app%3D138%26f%3DJPEG.webp
---

## 重复造轮子

为什么要重复造轮子？这是在写这个库之前，我一直在思考的问题。最浅显的理由就是结合自身长期开发视觉类的SDK经验，开发过程中经常遇到跨平台、跨工具链、跨设备、包冲突、性能瓶颈等等非常多的问题。这些问题说白了不论是集成OpenCV、Mobile-OpenCV还是dlib，只要多花点时间，总能解决。

但是真的让我决定重复造轮子的主要原因还是在于想动手实现一些自己想要的功能，比如常见的几何变换、图像处理、以及一些常用的算法。还有就是在几次与客户对接项目的过程中，出现了客户使用OpenCV作为依赖库的时候，出现了太多系统相关的符号问题，或许是系统链接的库与要求的版本有较多冲突，导致提高了很多本来没有必要的沟通成本。

通过对于我开发过的SDK的总结，我大部分的SDK都在解决以下场景的问题：

1. 跨平台，需要支持iOS、Android、Linux、macOS等系统；
2. 有多种嵌入式设备，需要支持多种芯片架构，如ARM、x86等；
3. 需要支持的嵌入式设备品牌较多，如Rockchip、HiSilicon、Amlogic、Allwinner、Qualcomm等等；
4. 需要足够的轻量化，不能占用太多内存和CPU资源和存储空间；
5. 需要支持多种图像格式，如RGB、BGR、YUV、GRAY等；
6. 项目需要有足够高的可移植性，依赖太多第三方库，导致项目变得非常臃肿；

鉴于以上问题，我决定重复造轮子，设计一个轻量化的计算机视觉库，满足以上需求。

## 对现有的一些库的总结

对于现有的计算机视觉库，最常见的就是OpenCV，从各个方面来说使用OpenCV作为CV类项目的依赖是一个最佳的选择，因为他所包含的算法库非常丰富，社区人员也非常多，遇到问题可以快速解决。

通过这几年的开发经验，日常开发项目都是使用OpenCV作为依赖库搭配一些推理框架所提供的CV库来搭配完成任务，比如NCNN、MNN或者Rockchip等一些芯片厂商都会提供一些支持自家设备可加速图像处理的接口如RGA等等。

对于以上这些库，通过个人习惯而言，在CPU场景下或者说通用场景下的一些图像预处理我最喜欢使用MNN的ImageProcess来完成，因为MNN的ImageProcess支持多种图像格式，并且支持多种图像处理操作，比如旋转、裁剪、缩放、翻转、亮度、对比度、饱和度、色调等并且支持图像的编解码如YUV、RGB、BGR、GRAY等。尤其是MNNImageProcess的实现逻辑采用自定义优化的Tensor作为基本数据结构，它是移植自Android 系统使用的Skia引擎，使用TransformMatrix配置图像变换的几何信息，再通过pipeline的形式实现从SRC到DST的图像变换，对于工程而言这是一种很优秀的图像变换逻辑，可以很简洁高效的完成图像变换，开发者只需要关心如何构建几何变换矩阵即可。

## 设计思路

我的设计思路非常简单，就是尽量按照OpenCV的一些常见接口来设计基本的图像处理接口，再按照MNN的ImageProcess的实现逻辑来设计图像变换的接口，并且除了基本的几何变换以外，还需要支持一些图像编解码的处理。同时在基本的图像处理接口中，我需要将接口进行抽象设计，我们把这层代码定义为Base，我们需要采用不同的backend来实现具体的功能，除了手动实现的backend以外，我们还需要支持使用OpenCV的backend，原因是：

1. 在CPU场景下，可以提供两种backend，如果用户有特殊需求可以切换到OpenCV的backend;
2. 方便我们在自己实现基础图像算法的同时可以进行单元测试，以OpenCV的实现作为参考基准，可以很方便的验证我们实现的正确性;
3. 为后期其他backend的接入提供基础，比如Rockchip的RGA、HiSilicon的ISP等等，这些对于跨平台、跨设备、跨芯片的场景非常有必要的；

## 基本算法设计

为了方便后续的扩展，我们统一采用与MNN的ImageProcess的思路，采用2D的图像变换矩阵来完成图像变换，图像变换是计算机图形学中的基础操作，通过矩阵运算可以实现各种空间变换。下面是常见的图像变换矩阵及其计算公式。

### 2D 变换矩阵

#### 1. 平移变换 (Translation)

将点 $(x, y)$ 平移 $(t_x, t_y)$ 个单位：

$$
\begin{bmatrix} x' \\ y' \\ 1 \end{bmatrix} = 
\begin{bmatrix} 
1 & 0 & t_x \\
0 & 1 & t_y \\
0 & 0 & 1
\end{bmatrix}
\begin{bmatrix} x \\ y \\ 1 \end{bmatrix}
$$

#### 2. 缩放变换 (Scaling)

将点 $(x, y)$ 按因子 $(s_x, s_y)$ 缩放：

$$
\begin{bmatrix} x' \\ y' \\ 1 \end{bmatrix} = 
\begin{bmatrix} 
s_x & 0 & 0 \\
0 & s_y & 0 \\
0 & 0 & 1
\end{bmatrix}
\begin{bmatrix} x \\ y \\ 1 \end{bmatrix}
$$

#### 3. 旋转变换 (Rotation)

将点 $(x, y)$ 绕原点逆时针旋转 $\theta$ 角度：

$$
\begin{bmatrix} x' \\ y' \\ 1 \end{bmatrix} = 
\begin{bmatrix} 
\cos\theta & -\sin\theta & 0 \\
\sin\theta & \cos\theta & 0 \\
0 & 0 & 1
\end{bmatrix}
\begin{bmatrix} x \\ y \\ 1 \end{bmatrix}
$$

#### 4. 切变变换 (Shear)

水平切变：

$$
\begin{bmatrix} x' \\ y' \\ 1 \end{bmatrix} = 
\begin{bmatrix} 
1 & \tan\alpha & 0 \\
0 & 1 & 0 \\
0 & 0 & 1
\end{bmatrix}
\begin{bmatrix} x \\ y \\ 1 \end{bmatrix}
$$

垂直切变：

$$
\begin{bmatrix} x' \\ y' \\ 1 \end{bmatrix} = 
\begin{bmatrix} 
1 & 0 & 0 \\
\tan\beta & 1 & 0 \\
0 & 0 & 1
\end{bmatrix}
\begin{bmatrix} x \\ y \\ 1 \end{bmatrix}
$$

#### 5. 反射变换 (Reflection)

X轴反射：

$$
\begin{bmatrix} x' \\ y' \\ 1 \end{bmatrix} = 
\begin{bmatrix} 
1 & 0 & 0 \\
0 & -1 & 0 \\
0 & 0 & 1
\end{bmatrix}
\begin{bmatrix} x \\ y \\ 1 \end{bmatrix}
$$

Y轴反射：

$$
\begin{bmatrix} x' \\ y' \\ 1 \end{bmatrix} = 
\begin{bmatrix} 
-1 & 0 & 0 \\
0 & 1 & 0 \\
0 & 0 & 1
\end{bmatrix}
\begin{bmatrix} x \\ y \\ 1 \end{bmatrix}
$$

原点反射：

$$
\begin{bmatrix} x' \\ y' \\ 1 \end{bmatrix} = 
\begin{bmatrix} 
-1 & 0 & 0 \\
0 & -1 & 0 \\
0 & 0 & 1
\end{bmatrix}
\begin{bmatrix} x \\ y \\ 1 \end{bmatrix}
$$

### 3D 变换矩阵

#### 1. 平移变换 (Translation)

将点 $(x, y, z)$ 平移 $(t_x, t_y, t_z)$ 个单位：

$$
\begin{bmatrix} x' \\ y' \\ z' \\ 1 \end{bmatrix} = 
\begin{bmatrix} 
1 & 0 & 0 & t_x \\
0 & 1 & 0 & t_y \\
0 & 0 & 1 & t_z \\
0 & 0 & 0 & 1
\end{bmatrix}
\begin{bmatrix} x \\ y \\ z \\ 1 \end{bmatrix}
$$

#### 2. 缩放变换 (Scaling)

将点 $(x, y, z)$ 按因子 $(s_x, s_y, s_z)$ 缩放：

$$
\begin{bmatrix} x' \\ y' \\ z' \\ 1 \end{bmatrix} = 
\begin{bmatrix} 
s_x & 0 & 0 & 0 \\
0 & s_y & 0 & 0 \\
0 & 0 & s_z & 0 \\
0 & 0 & 0 & 1
\end{bmatrix}
\begin{bmatrix} x \\ y \\ z \\ 1 \end{bmatrix}
$$

#### 3. 旋转变换 (Rotation)

绕X轴旋转 $\theta$ 角度：

$$
\begin{bmatrix} x' \\ y' \\ z' \\ 1 \end{bmatrix} = 
\begin{bmatrix} 
1 & 0 & 0 & 0 \\
0 & \cos\theta & -\sin\theta & 0 \\
0 & \sin\theta & \cos\theta & 0 \\
0 & 0 & 0 & 1
\end{bmatrix}
\begin{bmatrix} x \\ y \\ z \\ 1 \end{bmatrix}
$$

绕Y轴旋转 $\theta$ 角度：

$$
\begin{bmatrix} x' \\ y' \\ z' \\ 1 \end{bmatrix} = 
\begin{bmatrix} 
\cos\theta & 0 & \sin\theta & 0 \\
0 & 1 & 0 & 0 \\
-\sin\theta & 0 & \cos\theta & 0 \\
0 & 0 & 0 & 1
\end{bmatrix}
\begin{bmatrix} x \\ y \\ z \\ 1 \end{bmatrix}
$$

绕Z轴旋转 $\theta$ 角度：

$$
\begin{bmatrix} x' \\ y' \\ z' \\ 1 \end{bmatrix} = 
\begin{bmatrix} 
\cos\theta & -\sin\theta & 0 & 0 \\
\sin\theta & \cos\theta & 0 & 0 \\
0 & 0 & 1 & 0 \\
0 & 0 & 0 & 1
\end{bmatrix}
\begin{bmatrix} x \\ y \\ z \\ 1 \end{bmatrix}
$$

#### 4. 绕任意轴旋转

绕单位向量 $(u_x, u_y, u_z)$ 旋转 $\theta$ 角度：

$$
R = 
\begin{bmatrix} 
u_x^2(1-\cos\theta)+\cos\theta & u_xu_y(1-\cos\theta)-u_z\sin\theta & u_xu_z(1-\cos\theta)+u_y\sin\theta & 0 \\
u_xu_y(1-\cos\theta)+u_z\sin\theta & u_y^2(1-\cos\theta)+\cos\theta & u_yu_z(1-\cos\theta)-u_x\sin\theta & 0 \\
u_xu_z(1-\cos\theta)-u_y\sin\theta & u_yu_z(1-\cos\theta)+u_x\sin\theta & u_z^2(1-\cos\theta)+\cos\theta & 0 \\
0 & 0 & 0 & 1
\end{bmatrix}
$$

### 投影变换矩阵

#### 1. 正交投影 (Orthographic Projection)

将坐标从视图空间 $(x, y, z)$ 映射到标准化设备坐标 $(x', y', z')$：

$$
\begin{bmatrix} x' \\ y' \\ z' \\ 1 \end{bmatrix} = 
\begin{bmatrix} 
\frac{2}{r-l} & 0 & 0 & -\frac{r+l}{r-l} \\
0 & \frac{2}{t-b} & 0 & -\frac{t+b}{t-b} \\
0 & 0 & \frac{-2}{f-n} & -\frac{f+n}{f-n} \\
0 & 0 & 0 & 1
\end{bmatrix}
\begin{bmatrix} x \\ y \\ z \\ 1 \end{bmatrix}
$$

其中 $l, r$ 是左右边界，$b, t$ 是底部和顶部边界，$n, f$ 是近平面和远平面距离。

#### 2. 透视投影 (Perspective Projection)

$$
\begin{bmatrix} x' \\ y' \\ z' \\ 1 \end{bmatrix} = 
\begin{bmatrix} 
\frac{2n}{r-l} & 0 & \frac{r+l}{r-l} & 0 \\
0 & \frac{2n}{t-b} & \frac{t+b}{t-b} & 0 \\
0 & 0 & -\frac{f+n}{f-n} & -\frac{2fn}{f-n} \\
0 & 0 & -1 & 0
\end{bmatrix}
\begin{bmatrix} x \\ y \\ z \\ 1 \end{bmatrix}
$$

#### 3. 视图变换 (View Transformation)

将世界坐标系转换为相机坐标系：

$$
V = 
\begin{bmatrix} 
r_x & r_y & r_z & -\mathbf{r}\cdot\mathbf{e} \\
u_x & u_y & u_z & -\mathbf{u}\cdot\mathbf{e} \\
-f_x & -f_y & -f_z & \mathbf{f}\cdot\mathbf{e} \\
0 & 0 & 0 & 1
\end{bmatrix}
$$

其中 $\mathbf{r}$, $\mathbf{u}$, $\mathbf{f}$ 分别是相机坐标系的右、上、前方向向量，$\mathbf{e}$ 是相机位置。

#### 复合变换

多个变换矩阵可以通过矩阵乘法组合成一个变换：

$$
M = M_1 \times M_2 \times \cdots \times M_n
$$

注意变换的顺序是从右到左应用，即先应用 $M_n$，最后应用 $M_1$。

#### 逆变换

对于变换矩阵 $M$，其逆变换为 $M^{-1}$，满足：

$$
M \times M^{-1} = M^{-1} \times M = I
$$

其中 $I$ 是单位矩阵。

#### 刚体变换

刚体变换保持距离和角度不变，包括旋转和平移。形式为：

$$
\begin{bmatrix} 
R & T \\
0 & 1
\end{bmatrix}
$$

其中 $R$ 是旋转矩阵（正交矩阵），$T$ 是平移向量。