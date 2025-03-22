---
title: "Inspireface Runs on Rockchip Devices"
date: 2025-03-21 10:21:41 +/-0800
categories: [InspireFace]
tags: [Face,Computer Vision, EN]     # TAG names should always be lowercase
math: true
image: https://tunm-resource.oss-cn-hongkong.aliyuncs.com/blogs_box/logo04_s.png
---
## Rockchip Device Compatibility Status

InspireFace has been adapted for several mainstream Rockchip devices in the market. The main adaptation content includes RKNPU neural network acceleration inference solution, RGA image acceleration processing, and related code optimizations for compilers.

As of March 21, 2025, we have adapted the device platforms as shown in the following table. Among them, the RK devices that have been compiled, released, and tested include: **RV1109/RV1126** series, RV1103/RV1106 series, **RK3566/RK3568** series, and **RK3588**. Although RK continues to introduce various new chip models, some chips are actually compatible at the software level, such as using similar versions of NPU solutions and drivers, the same version of CPU architecture, and identical compilation chain tools, etc. For example, testing has shown that the InspireFace-SDK for the RK356X series and RK3588 is compatible. Therefore, in the future, we will try to minimize SDK coupling for Rockchip device adaptations and use compatible solutions for releases whenever possible.

| No. | Platform | Architecture<sup><br/>(CPU) | Device<sup><br/>(Special) | **Supported** | Passed Tests | Release<sup><br/>(Online) |
| ------- | -------------------- | --------------------- | -------------------------- | :-----------: | :----------------: | :----------------: |
| 1       | **Linux**<sup><br/>(CPU)      | ARMv7                 | -                          | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![build](https://img.shields.io/github/actions/workflow/status/HyperInspire/InspireFace/release-sdks.yaml?label=✓&labelColor=success&color=success&failedLabel=✗&failedColor=critical&logo=github&logoColor=white)](https://github.com/HyperInspire/InspireFace/actions/workflows/release-sdks.yaml) |
| 2       |                      | ARMv8                 | -                          | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![build](https://img.shields.io/github/actions/workflow/status/HyperInspire/InspireFace/release-sdks.yaml?label=✓&labelColor=success&color=success&failedLabel=✗&failedColor=critical&logo=github&logoColor=white)](https://github.com/HyperInspire/InspireFace/actions/workflows/release-sdks.yaml) |
| 3       |                      | x86/x86_64            | -                          | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![build](https://img.shields.io/github/actions/workflow/status/HyperInspire/InspireFace/release-sdks.yaml?label=✓&labelColor=success&color=success&failedLabel=✗&failedColor=critical&logo=github&logoColor=white)](https://github.com/HyperInspire/InspireFace/actions/workflows/release-sdks.yaml) |
| 4       | **Linux**<sup><br/>(Rockchip) | ARMv7                 | RV1109/RV1126              | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![build](https://img.shields.io/github/actions/workflow/status/HyperInspire/InspireFace/release-sdks.yaml?label=✓&labelColor=success&color=success&failedLabel=✗&failedColor=critical&logo=github&logoColor=white)](https://github.com/HyperInspire/InspireFace/actions/workflows/release-sdks.yaml) |
| 5 | | ARMv7 | RV1103/RV1106 | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![build](https://img.shields.io/github/actions/workflow/status/HyperInspire/InspireFace/release-sdks.yaml?label=✓&labelColor=success&color=success&failedLabel=✗&failedColor=critical&logo=github&logoColor=white)](https://github.com/HyperInspire/InspireFace/actions/workflows/release-sdks.yaml) |
| 6 | | ARMv8 | RK3566/RK3568 | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![build](https://img.shields.io/github/actions/workflow/status/HyperInspire/InspireFace/release-sdks.yaml?label=✓&labelColor=success&color=success&failedLabel=✗&failedColor=critical&logo=github&logoColor=white)](https://github.com/HyperInspire/InspireFace/actions/workflows/release-sdks.yaml) |
| 7 | | ARMv8 | RK3588 | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | - | [![build](https://img.shields.io/github/actions/workflow/status/HyperInspire/InspireFace/release-sdks.yaml?label=✓&labelColor=success&color=success&failedLabel=✗&failedColor=critical&logo=github&logoColor=white)](https://github.com/HyperInspire/InspireFace/actions/workflows/release-sdks.yaml) |
| 8      | **Linux**<sup><br/>(MNN_CUDA) | x86/x86_64            | NVIDIA-GPU          | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | - |
| 9      | **Linux**<sup><br/>(CUDA) | x86/x86_64            | NVIDIA-GPU          | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![build](https://img.shields.io/github/actions/workflow/status/HyperInspire/InspireFace/release-sdks.yaml?label=✓&labelColor=success&color=success&failedLabel=✗&failedColor=critical&logo=github&logoColor=white)](https://github.com/HyperInspire/InspireFace/actions/workflows/release-sdks.yaml) |
| 10      | **MacOS**           | Intel       | CPU/Metal/**ANE** | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![build](https://img.shields.io/github/actions/workflow/status/HyperInspire/InspireFace/release-sdks.yaml?label=✓&labelColor=success&color=success&failedLabel=✗&failedColor=critical&logo=github&logoColor=white)](https://github.com/HyperInspire/InspireFace/actions/workflows/release-sdks.yaml) |
| 11   |                      | Apple Silicon         | -                          | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![build](https://img.shields.io/github/actions/workflow/status/HyperInspire/InspireFace/release-sdks.yaml?label=✓&labelColor=success&color=success&failedLabel=✗&failedColor=critical&logo=github&logoColor=white)](https://github.com/HyperInspire/InspireFace/actions/workflows/release-sdks.yaml) |
| 12     | **iOS**              | ARM                   | CPU/Metal/**ANE**         | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![build](https://img.shields.io/github/actions/workflow/status/HyperInspire/InspireFace/release-sdks.yaml?label=✓&labelColor=success&color=success&failedLabel=✗&failedColor=critical&logo=github&logoColor=white)](https://github.com/HyperInspire/InspireFace/actions/workflows/release-sdks.yaml) |
| 13     | **Android**          | ARMv7                 | -                          | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![build](https://img.shields.io/github/actions/workflow/status/HyperInspire/InspireFace/release-sdks.yaml?label=✓&labelColor=success&color=success&failedLabel=✗&failedColor=critical&logo=github&logoColor=white)](https://github.com/HyperInspire/InspireFace/actions/workflows/release-sdks.yaml) |
| 14     |                      | ARMv8                 | -                          | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![build](https://img.shields.io/github/actions/workflow/status/HyperInspire/InspireFace/release-sdks.yaml?label=✓&labelColor=success&color=success&failedLabel=✗&failedColor=critical&logo=github&logoColor=white)](https://github.com/HyperInspire/InspireFace/actions/workflows/release-sdks.yaml) |
| 15 | **Android**<sup><br/>(Rockchip) | ARMv8 | RK3566/RK3568 | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![build](https://img.shields.io/github/actions/workflow/status/HyperInspire/InspireFace/release-sdks.yaml?label=✓&labelColor=success&color=success&failedLabel=✗&failedColor=critical&logo=github&logoColor=white)](https://github.com/HyperInspire/InspireFace/actions/workflows/release-sdks.yaml) |
| 16 |  | ARMv8 | RK3588 | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![](https://img.shields.io/badge/%E2%9C%93-green)](#) | [![build](https://img.shields.io/github/actions/workflow/status/HyperInspire/InspireFace/release-sdks.yaml?label=✓&labelColor=success&color=success&failedLabel=✗&failedColor=critical&logo=github&logoColor=white)](https://github.com/HyperInspire/InspireFace/actions/workflows/release-sdks.yaml) |

## How to Obtain the SDK

You can download the relevant SDK files from InspireFace's [release page](https://github.com/HyperInspire/InspireFace/releases), which are published only after testing.

If you need to modify the source code of InspireFace, you will need to compile the SDK yourself. Taking RK356X and RK3588 devices as examples, you can refer to the build script **command/build_cross_rk356x_rk3588_aarch64.sh** in the root directory:

```bash
cmake -DCMAKE_SYSTEM_NAME=Linux \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_SYSTEM_VERSION=1 \
  -DCMAKE_SYSTEM_PROCESSOR=aarch64 \
  -DCMAKE_C_COMPILER=$ARM_CROSS_COMPILE_TOOLCHAIN/bin/aarch64-linux-gnu-gcc \
  -DCMAKE_CXX_COMPILER=$ARM_CROSS_COMPILE_TOOLCHAIN/bin/aarch64-linux-gnu-g++ \
  -DCMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS} -flax-vector-conversions" \
  -DTARGET_PLATFORM=armlinux \
  -DISF_BUILD_LINUX_AARCH64=ON \
  -DISF_BUILD_LINUX_ARM7=OFF \
  -DMNN_SEP_BUILD=off \
  -DISF_ENABLE_RKNN=ON \
  -DISF_RK_DEVICE_TYPE=RK356X \
  -DISF_RKNPU_MAJOR=rknpu2 \
  -DISF_RK_COMPILER_TYPE=aarch64 \
  -DISF_ENABLE_RGA=ON \
  -DISF_ENABLE_COST_TIME=OFF \
  -DISF_BUILD_WITH_SAMPLE=OFF \
  -DISF_BUILD_WITH_TEST=OFF \
  -DISF_ENABLE_BENCHMARK=OFF \
  -DISF_ENABLE_USE_LFW_DATA=OFF \
  -DISF_ENABLE_TEST_EVALUATION=OFF \
  -DISF_BUILD_SHARED_LIBS=OFF ${SCRIPT_DIR}

make -j4
make install
```

The toolchain directory variable **ARM_CROSS_COMPILE_TOOLCHAIN** needs to be specified in advance, for example:

```bash
export ARM_CROSS_COMPILE_TOOLCHAIN=/host/software/gcc-linaro-6.3.1-2017.05-x86_64_aarch64-linux-gnu
```

## About RGA Adaptation

For Linux systems, we have tested and confirmed that RGA acceleration is available on the following devices:

- RV1103/RV1106
- RK3566/RK3568
- RK3588

For Android systems, we encountered some linking errors during the compilation with RGA adaptation. We are currently looking for more solutions, so for the current version of Android, you need to disable the RGA switch to compile successfully:

```bash
cmake -DISF_ENABLE_RGA=OFF ..
```
