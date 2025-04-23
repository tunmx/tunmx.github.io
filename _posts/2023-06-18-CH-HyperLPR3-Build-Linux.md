---
title: "HyperLPR3车牌识别-Linux/MacOS使用：C/C++库编译"
date: 2023-06-18 02:54:53 +/-0800
categories: [HyperLPR]
tags: [CH]     # TAG names should always be lowercase
math: true
image: https://inspireface-1259028827.cos.ap-singapore.myqcloud.com/blogs_box/hyperlpr_logo_cl.png
---


## 简介

HyperLPR在2023年初已经更新到了v3的版本，该版本与先前的版本一样都是用于识别中文车牌的开源图像算法项目，最新的版本的源码可从github中提取：[https://github.com/szad670401/HyperLPR](https://github.com/szad670401/HyperLPR)

### 支持多种类型车牌

![](https://inspireface-1259028827.cos.ap-singapore.myqcloud.com/blogs_box/f08da74501e9f3c6e40d0b33c32769d5.png)

## Linux C/C++库说明

项目支持编译出C/C++项目使用的动态链接库，暂时仅支持Linux、MacOS等这类*nix系列的系统。库采用CAPI的接口形式进行调用，仅需一个头文件+动态库即可。工程需要使用CMake进行编译。

## 准备工作

在编译库之前需要提前准备好以编译的工作环境，需要依赖如下：

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

### 安装CMake

CMake安装方式众多，Ubuntu用户可使用apt快捷安装：

```bash
sudo apt-get install cmake
```

MacOS可使用brew进行快捷安装：

```bash
sudo brew install cmake
```

如果快捷安装失败也可到CMake官方下载最新版自行编译安装。

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

## 动态库快捷编译

准备好以上工作后，执行编译脚本即可开始编译：

```
# 执行编译脚本
sh command/build_release_linux_share.sh
```

编译后的相关物料放置于根目录下**build/linux/install/hyperlpr3**中，其中包含：

- include 头文件

- lib 动态库路径

- resource 包含测试图片与模型等静态资源

按需取走需要的文件即可

## 使用Demo

编译好动态链接库后，我们提供了一个使用Demo，即根目录下的Prj-Linux文件夹，在编译完成上面的动态库后即可进入到该目录下进行测试，该demo仅体现SDK最简单的使用方式，代码如下：

```cpp
#include <iostream>
#include "hyper_lpr_sdk.h"
#include "opencv2/opencv.hpp"
 
static const std::vector<std::string> TYPES = {"蓝牌", "黄牌单层", "白牌单层", "绿牌新能源", "黑牌港澳", "香港单层", "香港双层", "澳门单层", "澳门双层", "黄牌双层"};
 
 
int main(int argc, char **argv) {
    char *model_path = argv[1];
    char *image_path = argv[2];
    // 读取图像
    cv::Mat image = cv::imread(image_path);
    // 创建ImageData
    HLPR_ImageData data = {0};
    data.data = image.ptr<uint8_t>(0);      // 设置图像数据流
    data.width = image.cols;                   // 设置图像宽
    data.height = image.rows;                  // 设置图像高
    data.format = STREAM_BGR;                  // 设置当前图像编码格式
    data.rotation = CAMERA_ROTATION_0;         // 设置当前图像转角
    // 创建数据Buffer
    P_HLPR_DataBuffer buffer = HLPR_CreateDataBuffer(&data);
 
    // 配置车牌识别参数
    HLPR_ContextConfiguration configuration = {0};
    configuration.models_path = model_path;         // 模型文件夹路径
    configuration.max_num = 5;                      // 最大识别车牌数量
    configuration.det_level = DETECT_LEVEL_LOW;     // 检测器等级
    configuration.use_half = false;
    configuration.nms_threshold = 0.5f;             // 非极大值抑制置信度阈值
    configuration.rec_confidence_threshold = 0.5f;  // 车牌号文本阈值
    configuration.box_conf_threshold = 0.30f;       // 检测器阈值
    configuration.threads = 1;
    // 实例化车牌识别算法Context
    P_HLPR_Context ctx = HLPR_CreateContext(&configuration);
    // 查询实例化状态
    HREESULT ret = HLPR_ContextQueryStatus(ctx);
    if (ret != HResultCode::Ok) {
        printf("create error.\n");
        return -1;
    }
    HLPR_PlateResultList results = {0};
    // 执行车牌识别算法
    HLPR_ContextUpdateStream(ctx, buffer, &results);
 
    for (int i = 0; i < results.plate_size; ++i) {
        // 解析识别后的数据
        std::string type;
        if (results.plates[i].type == HLPR_PlateType::PLATE_TYPE_UNKNOWN) {
            type = "未知";
        } else {
            type = TYPES[results.plates[i].type];
        }
 
        printf("<%d> %s, %s, %f\n", i + 1, type.c_str(),
               results.plates[i].code, results.plates[i].text_confidence);
    }
 
    // 销毁Buffer
    HLPR_ReleaseDataBuffer(buffer);
    // 销毁Context
    HLPR_ReleaseContext(ctx);
 
    return 0;
}
```

执行命令即可编译：
```bash
# 进入到子工程demo
cd Prj-Linux
# 执行编译脚本
sh build.sh
# 进入目录
cd build/
# 编译好PlateRecDemo程序后 传入模型文件夹路径和需要预测的图像执行程序
./PlateRecDemo ../hyperlpr3/resource/models/r2_mobile ../hyperlpr3/resource/images/test_img.jpg
```

## 单元测试编译

如果需要验证代码、库与模型之间构成是否正确，可使用项目中自带的单元测试进行编译和测试:

```bash
# 创建编译目录 并进入
mkdir build_test && cd build_test
# 编译单元测试程序
cmake -DBUILD_TEST=ON .. && make -j8
```

编译完成后需要执行单元测试程序，需要将resource链接到程序同级目录下再执行测试用例:

```
# 使用链接的方式设置resource目录
ln -s ../resource .
# 执行测试
./UnitTest
```

## 帮助

以上为HyperLPR3的C/C++快速上手，需要获取其他的帮助，请移步到项目地址：[https://github.com/szad670401/HyperLPR](https://github.com/szad670401/HyperLPR)

