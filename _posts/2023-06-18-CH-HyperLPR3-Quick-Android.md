---
title: "HyperLPR3车牌识别-Android-SDK光速部署与使用"
date: 2023-06-18 05:11:24 +/-0800
categories: [HyperLPR]
tags: [CH, Computer Vision]     # TAG names should always be lowercase
math: true
image: https://inspireface-1259028827.cos.ap-singapore.myqcloud.com/blogs_box/hyperlpr_logo_cl.png
---


## 简介

HyperLPR在2023年初已经更新到了v3的版本，该版本与先前的版本一样都是用于识别中文车牌的开源图像算法项目，最新的版本的源码可从github中提取：[https://github.com/szad670401/HyperLPR](https://github.com/szad670401/HyperLPR)

## HyperLPR-Android-SDK for JitPack

HyperLPR3的官方源码已经提供在Android平台下使用该项目SDK的方法。Android SDK for HyperLPR3的组成部分主要为：HyperLPR3的Android工程模块、算法动态链接库、资源文件三个部分组成。但是对于一些不需要编译的同学，HyperLPR官方也提供了使用Jitpack作为依赖的导入方式，可以快速导入车牌识别SDK并进行快速部署使用到项目中。

## JitPack依赖

在你的AndroidStudio工程下，找到你的settings.gradle文件，并将jitpack依赖放入：

```gradle
repositories {
    ...
    maven { url 'https://jitpack.io' }
}
```

如图所示：

![](https://inspireface-1259028827.cos.ap-singapore.myqcloud.com/blogs_box/b2bf5d45f78c955e2be2efb1a5b8de15.png)

## 从项目中导入hyperlpr-android-sdk

在你需要引入的工程所对应的build.gradle文件中引入hyperlpr-android-sdk的依赖：

```gradle
dependencies {
    ...
    implementation 'com.github.HyperInspire:hyperlpr3-android-sdk:1.0.3'
}
```

如下图在app的工程中引入：

![](https://inspireface-1259028827.cos.ap-singapore.myqcloud.com/blogs_box/8a7d6651bd3373c8f16ffed7ca3fdc77.png)

完成以上步骤后，点击Sync Now开始导入依赖包，由于依赖包有一定的体积，加上网络环境可能会较差，导包时间会较长：

![](https://inspireface-1259028827.cos.ap-singapore.myqcloud.com/blogs_box/e54fab2930c3770c399fb85216bbd945.png)

## 开始使用车牌识别SDK

当完成以上步骤后，即可在安卓工程中使用车牌识别SDK

### SDK初始化

在使用SDK时需要对SDK进行初始化配置，否则车牌识别算法会失效，初始化仅需也只能执行一次，通常推荐放置于程序运行时的主程序页面中进行注册。初始化需要传入一个Context和车牌识别参数对象HyperLPRParameter，参数需根据用户使用进行调整。

```java
// 车牌识别算法配置参数
HyperLPRParameter parameter = new HyperLPRParameter()
        .setDetLevel(HyperLPR3.DETECT_LEVEL_LOW)
        .setMaxNum(1)
        .setRecConfidenceThreshold(0.85f);
// 初始化(仅执行一次生效)
HyperLPR3.getInstance().init(this, parameter);
```

### 使用车牌识别函数

完成初始化后，即可使用车牌识别函数，这里以一个Bitmap图像作为示例进行调用：

```java
// 使用Bitmap作为图片参数进行车牌识别
Plate[] plates =  HyperLPR3.getInstance().plateRecognition(bitmap, HyperLPR3.CAMERA_ROTATION_0, HyperLPR3.STREAM_BGRA);
for (Plate plate: plates) {
    // 打印检测到的车牌号
    Log.i(TAG, plate.getCode());
}
```

如上所示，仅使用几句代码就可以实现车牌识别的部署与最快调试。

## 更多示例

如果以上的功能无法满足或帮助到你，我们在项目源工程中提供了一个更加丰富的使用案例：[Prj-Android](https://github.com/szad670401/HyperLPR/tree/master/Prj-Android)，你可以使用AndroidStudio打开并运行这个项目，项目中包含了图片识别车牌与实时识别车牌的案例，希望可以帮助到你。

## 直接体验

如果你需要直接体验HyperLPR安卓版本的Demo，官网已经提供好了APK，只需[扫码下载](http://fir.tunm.top/hyperlpr)即可安装

![https://inspireface-1259028827.cos.ap-singapore.myqcloud.com/blogs_box/7a37b9ed636db9d7aebf5306b13fabe7.jpeg](https://inspireface-1259028827.cos.ap-singapore.myqcloud.com/blogs_box/7a37b9ed636db9d7aebf5306b13fabe7.jpeg){: width="320" height="512" }

## 支持更丰富的车牌种类


![](https://inspireface-1259028827.cos.ap-singapore.myqcloud.com/blogs_box/f08da74501e9f3c6e40d0b33c32769d5.png)


## 帮助

以上为HyperLPR3的C/C++快速上手，需要获取其他的帮助，请移步到项目地址：[https://github.com/szad670401/HyperLPR](https://github.com/szad670401/HyperLPR)

