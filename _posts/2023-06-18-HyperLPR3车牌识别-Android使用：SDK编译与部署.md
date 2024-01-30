---
title: "HyperLPR3车牌识别-Android使用：SDK编译与部署"
date: 2023-06-18 12:12:51 +/-0800
categories: [HyperLPR]
tags: [CH]     # TAG names should always be lowercase
math: true
image: https://tunm-resource.oss-cn-hongkong.aliyuncs.com/blogs_box/hyperlpr_logo_cl.png
---


## 简介

HyperLPR在2023年初已经更新到了v3的版本，该版本与先前的版本一样都是用于识别中文车牌的开源图像算法项目，最新的版本的源码可从github中提取：[https://github.com/szad670401/HyperLPR](https://github.com/szad670401/HyperLPR)

## HyperLPR3 for Android-SDK说明

HyperLPR3的官方源码已经提供在Android平台下使用该项目SDK的方法。Android SDK for HyperLPR3的组成部分主要为：HyperLPR3的Android工程模块、算法动态链接库、资源文件三个部分组成。


### 支持多种类型车牌

![](https://tunm-resource.oss-cn-hongkong.aliyuncs.com/blogs_box/f08da74501e9f3c6e40d0b33c32769d5.png)

## 编译源码的准备工作

目前官方提供的编译方式仅支持在Linux和MacOS下进行交叉编译，在编译库之前需要提前准备好以编译的工作环境，需要依赖如下：


- CMake（3.10 以上）

- OpenCV (4.20以上)

    - 编译对应的平台如android、ios、linux都需要采用各自平台支持的OpenCV-SDK

- MNN (2.0.0以上)

- C++编译器

    - GCC或Clang皆可 (macOS无需另外安装，Xcode自带)

        - GCC推荐版本4.9以上

            - 在某些发行版上GCC (GNU C编译器)和G++(GNU C++编译器是分开安装的)。

            - 同样以Ubuntu为例，需要分别安装 gcc 和 g++

        - Clang 推荐版本3.9以上

- Catch2（仅编译TestCase需要）

### 拉取项目

从github或gitee中拉取最新版的HyperLPR工程到本地:

```bash
# 从github拉取
git clone https://github.com/szad670401/HyperLPR.git
```

网络不太稳的同学可以使用gitee拉取项目:

```bash
# 从gitee拉取
git clone https://gitee.com/tunmx/HyperLPR.git
```

### 安装NDK

交叉编译Android版本需要下载并安装NDK，如果你的计算机里已经有NDK版本了也可优先尝试你的版本是否能正常编译和正常使用，如果你没有安装过NDK，那优先推荐使用21版。NDK可从官网发布页面进行下载：[NDK下载](https://github.com/android/ndk/wiki/Unsupported-Downloads)

选择下载支持你宿主机的NDK后，将其解压放置于你硬盘中的某个路径后即可，放好NDK后需要配置系统环境变量**ANDROID_NDK**：

终端临时配置：

```bash
# 临时配置，重启终端会失效
export ANDROID_NDK=/YOUR_PATH/android-ndk-r21e-linux/
```

- Linux用户：

```bash
# 打开用户目录下的bashrc
vim ~/.bashrc
# 并写入
ANDROID_NDK=/YOUR_PATH/android-ndk-r21e-linux/
 
# 也可打开用户目录profile进行配置
vim ~/.profile
# 并写入
ANDROID_NDK=/YOUR_PATH/android-ndk-r21e-linux/
```

- MacOS用户：

```bash
# 打开用户目录下的bash_profile
vim ~/.bash_profile
# 并写入
ANDROID_NDK=/YOUR_PATH/android-ndk-r21e-linux/
 
# 如果使用zsh可直接在zshrc文件下写入
vim ~/.zshrc
# 并写入
ANDROID_NDK=/YOUR_PATH/android-ndk-r21e-linux/
```

### 安装依赖库

为了节约编译三方依赖库的时间，这边提供了一份已经编译好的库，提供的库均为功能相对标准的库，如有其他功能的需求，请自行调整参数重新编译。![百度网盘](https://pan.baidu.com/s/1zfP2MSsG1jgxB_MjvpwZJQ) 密码：eu31，下载后将其拷贝或链接到HyperLPR根目录下，与CMakeLists.txt同级即可：
```bash
HyperLPR/
├── 3rdparty_hyper_inspire_op # 放这里
├── CMakeLists.txt
├── LICENSE
├── Prj-Android
├── Prj-Linux
├── Prj-Python
├── README.md
├── build
├── cmake-build-debug
├── command
├── cpp
├── docs
├── images
└── resource
```

## 执行交叉编译源码

准备好以上工作后，即可执行Android版本动态链接库的交叉编译工作，为了方便各位同学，官方项目已提供快捷编译的脚本，在确保以上步骤均无问题的情况下直接执行脚本即可完成编译：

```bash
# 执行编译脚本
sh command/build_release_linux_share.sh
```

编译完成后产生的文件会放置于根目录下的build/release_android文件夹中，其中包含：

- arm64-v8a文件夹：Android Armv8版本的动态库

- armeabi-v7a文件夹：Android Armv7版本的动态库

## 准备SDK模块

HyperLPR3已经有官方提供的Android版的SDK以及使用Demo工程，可以从地址：[https://github.com/HyperInspire/hyperlpr3-android-sdk.git](https://github.com/HyperInspire/hyperlpr3-android-sdk.git)中拉取。并执行完上面的交叉编译后，我们需要将编译后的动态库拷贝到hyperlpr3-android-sdk工程下的SDK模块库目录下：

```bash
# 拉取工程
git clone https://github.com/HyperInspire/hyperlpr3-android-sdk.git
# 创建安卓工程中SDK模块的libs目录
mkdir hyperlpr3-android-sdk/hyperlpr3/libs/
# 从HyperLPR工程中拷贝编译后的动态库到libs文件夹中
cp -r YOUR_PATH/HyperLPR/build/release_android/arm* hyperlpr3-android-sdk/hyperlpr3/libs
```

拷贝完成后即完成SDK模块的准备工作.

## 构建Android-Demo工程

使用Android Studio打开**hyperlpr3-android-sdk**工程，使用gradle进行项目构建。工程中的hyperlpr模块为车牌识别的SDK，如需在其他工程中使用，拷贝并设置依赖即可。

点击Run即可构建工程并编译后部署到测试机上进行运行。

![https://tunm-resource.oss-cn-hongkong.aliyuncs.com/blogs_box/7a37b9ed636db9d7aebf5306b13fabe7.jpeg](https://tunm-resource.oss-cn-hongkong.aliyuncs.com/blogs_box/7a37b9ed636db9d7aebf5306b13fabe7.jpeg){: width="320" height="512" }

Demo有两种测试方式：实测后置相机采集与相册图片测试

## SDK代码快速使用

在导入SDK模块后，即可在其他工程使用SDK代码，接口相对简单：

```java
/**
 * Initialize the license plate recognition algorithm SDK
 *
 * @param context context
 * @param parameter Initialization parameter
 * */
void init(Context context, HyperLPRParameter parameter);
 
/**
 * License plate recognition interface.
 *
 * @param buf Image data buffer.
 * @param height Height of the image
 * @param width Width of the image
 * @param rotation Original data buffer rotation Angle
 * @param format Buffer data coded format
 * @return Resulting object array
 */
Plate[] plateRecognition(byte[] buf, int height, int width, int rotation, int format);
 
/**
 * License plate recognition interface.
 *
 * @param image Bitmap image
 * @param rotation Original data buffer rotation Angle
 * @param format Buffer data coded format
 * @return Resulting object array
 */
Plate[] plateRecognition(Bitmap image, int rotation, int format);
```

调用仅需实例化HyperLPR3对象即可，需要导入包:

```java
import com.hyperai.hyperlpr3.HyperLPR3;
import com.hyperai.hyperlpr3.bean.Parameter;
import com.hyperai.hyperlpr3.bean.Plate;
```

实例化对象与检测:

```java
// 车牌识别算法配置参数
HyperLPRParameter parameter = newHyperLPRParameter()
        .setDetLevel(HyperLPR3.DETECT_LEVEL_LOW)
        .setMaxNum(1)
        .setRecConfidenceThreshold(0.85f);
// 初始化(仅执行一次生效)
HyperLPR3.getInstance().init(this, parameter);
```

## SDK定义类型说明

在提供的SDK中，官方提供了一些类型以便于开发者进行二次开发调用：

```java
/** 四种情况的转角 */
public static final int CAMERA_ROTATION_0 = 0;
public static final int CAMERA_ROTATION_90 = 1;
public static final int CAMERA_ROTATION_180 = 2;
public static final int CAMERA_ROTATION_270 = 3;
 
/** 低开销检测模式 */
public static final int DETECT_LEVEL_LOW = 0;
/** 高开销检测模式 */
public static final int DETECT_LEVEL_HIGH = 1;
 
/** Image in RGB format - RGB排列格式的图像 */
public static final int STREAM_RGB = 0;
/** Image in BGR format (Opencv Mat default) - BGR排列格式的图像(OpenCV的Mat默认) */
public static final int STREAM_BGR = 1;
/** Image in RGB with alpha channel format -带alpha通道的RGB排列格式的图像 */
public static final int STREAM_RGBA = 2;
/** Image in BGR with alpha channel format -带alpha通道的BGR排列格式的图像 */
public static final int STREAM_BGRA = 3;
/** Image in YUV NV12 format - YUV NV12排列的图像格式 */
public static final int STREAM_YUV_NV12 = 4;
/** Image in YUV NV21 format - YUV NV21排列的图像格式 */
public static final int STREAM_YUV_NV21 = 5;
 
 
/** 未知车牌 */
public static final int PLATE_TYPE_UNKNOWN = -1;
/** 蓝牌 */
public static final int PLATE_TYPE_BLUE = 0;
/** 黄牌单层 */
public static final int PLATE_TYPE_YELLOW_SINGLE = 1;
/** 白牌单层 */
public static final int PLATE_TYPE_WHILE_SINGLE = 2;
/** 绿牌新能源 */
public static final int PLATE_TYPE_GREEN = 3;
/** 黑牌港澳 */
public static final int PLATE_TYPE_BLACK_HK_MACAO = 4;
/** 香港单层 */
public static final int PLATE_TYPE_HK_SINGLE = 5;
/** 香港双层 */
public static final int PLATE_TYPE_HK_DOUBLE = 6;
/** 澳门单层 */
public static final int PLATE_TYPE_MACAO_SINGLE = 7;
/** 澳门双层 */
public static final int PLATE_TYPE_MACAO_DOUBLE = 8;
/** 黄牌双层 */
public static final int PLATE_TYPE_YELLOW_DOUBLE = 9;
 
public static final String[] PLATE_TYPE_MAPS = {"蓝牌", "黄牌单层", "白牌单层", "绿牌新能源", "黑牌港澳", "香港单层", "香港双层", "澳门单层", "澳门双层", "黄牌双层"};
```

## 帮助

以上为HyperLPR3的C/C++快速上手，需要获取其他的帮助，请移步到项目地址：[https://github.com/szad670401/HyperLPR](https://github.com/szad670401/HyperLPR)

