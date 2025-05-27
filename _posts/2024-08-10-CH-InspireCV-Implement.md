---
title: "InspireCV-自定义轻量化CV库"
date: 2025-05-01 02:45:02 +/-0800
categories: [Programming]
tags: [Face, Computer Vision, CH]     # TAG names should always be lowercase
math: true
image: https://inspireface-1259028827.cos.ap-singapore.myqcloud.com/docs/feature/fbanner.jpg
---

## InspireCV的实现

**InspireCV** 的开发目的是通过用轻量级的、项目定制的视觉库替换 OpenCV 来减少 SDK 大小并避免依赖问题，为常用的视觉算法提供高级抽象接口。它具有灵活的后端架构，允许用户默认使用轻量级后端，同时也提供切换到更强大的 **OpenCV 后端** 的选项以获得增强的性能。


- 项目地址：[InspireCV](https://github.com/tunmx/InspireCV)

InspireCV在一些特殊的硬件会加入一些提升速度的尝试，例如使用**Rockchip RGA**的方式利用DMA加速2D图像的处理等，最终目的是让用户可以最轻的进行多平台移植CV项目，并同时兼顾性能和处理结果。


![InspireCV](https://inspireface-1259028827.cos.ap-singapore.myqcloud.com/docs/cv2.jpg)

<hr>

- 同时支持 OpenCV 和自定义 OKCV 后端
- 核心功能包括：
  - 基本图像处理操作
  - 几何基元（Point、Rect、Size）
  - 变换矩阵
  - 图像输入/输出
- 使用 OKCV 后端时依赖最小
- 可选的 OpenCV 集成用于调试和可视化

## 构建选项

### 后端选择

- `INSPIRECV_BACKEND_OPENCV`: 使用 OpenCV 作为后端（默认关闭）
- `INSPIRECV_BACKEND_OKCV_USE_OPENCV`: 在 OKCV 后端中启用 OpenCV 支持（默认关闭）
- `INSPIRECV_BACKEND_OKCV_USE_OPENCV_IO`: 在 OKCV 中使用 OpenCV 的图像 I/O（默认关闭）
- `INSPIRECV_BACKEND_OKCV_USE_OPENCV_GUI`: 在 OKCV 中使用 OpenCV 的 GUI 功能（默认关闭）

### 其他选项

- `INSPIRECV_BUILD_SHARED_LIBS`: 构建为共享库（默认关闭）
- `INSPIRECV_OKCV_BUILD_TESTS`: 构建测试套件（默认开启）
- `INSPIRECV_OKCV_BUILD_SAMPLE`: 构建示例应用程序（默认开启）

### 依赖

必需：

- CMake 3.10+
- Eigen3
- C++14 编译器

可选：

- OpenCV（如果使用 OpenCV 后端或 OKCV 中的 OpenCV 功能则必需）

## 使用指南

### 图像输入/输出

图像可以通过多种方式从文件、缓冲区或其他来源加载。默认图像类型是 3 通道 **BGR** 图像，与 OpenCV 相同。

- **图像构造函数**

```cpp
// 从文件加载图像
// 以 3 通道加载（BGR，与 opencv 相同）
inspirecv::Image img = inspirecv::Image::Create("test_res/data/bulk/kun_cartoon_crop.jpg", 3);

// 其他加载方法

// 从缓冲区加载图像
uint8_t* buffer = ...;  // buffer 是指向图像数据的指针
bool is_alloc_mem = false;  // 如果为 true，将为图像数据分配内存，
                            // 建议使用 false 以指向原始数据避免复制
inspirecv::Image img = inspirecv::Image::Create(width, height, channel, buffer, is_alloc_mem);
```



- **图像保存和显示**

图像支持多种图像格式，包括 PNG、JPG、BMP 等。您可以将图像保存到文件。如果您想显示图像，必须依赖 OpenCV。

```cpp
// 将图像保存到文件
img.Write("output.jpg");

// 显示图像，警告：必须依赖 opencv
img.Show("input");
```



- **获取图像指针**

```cpp
// 获取图像数据指针
const uint8_t* ptr = img.Data();
```



### 图像处理

图像处理是 InspireCV 的核心功能。它提供了一组处理图像的函数。

以这张原始图像为例：

![KunKun](https://inspireface-1259028827.cos.ap-singapore.myqcloud.com/docs/kun_cartoon_crop.jpg)

功能包括：

- **转灰度**

```cpp
inspirecv::Image gray = img.ToGray();
```



![Gray Image](https://inspireface-1259028827.cos.ap-singapore.myqcloud.com/docs/cv/gray.jpg)

---

- **应用高斯模糊**

```cpp
inspirecv::Image blurred = img.GaussianBlur(3, 1.0);
```



![Blurred Image](https://inspireface-1259028827.cos.ap-singapore.myqcloud.com/docs/cv/blurred.jpg)

---

- **调整大小**

```cpp
auto scale = 0.35;
bool use_bilinear = true;
inspirecv::Image resized = img.Resize(img.Width() * scale, img.Height() * scale, use_bilinear);
```



![Resized Image](https://inspireface-1259028827.cos.ap-singapore.myqcloud.com/docs/cv/resized.jpg)

---

- **旋转**
    - 支持 90、180、270 度顺时针旋转

```cpp
inspirecv::Image rotated = img.Rotate90();
```



![Rotated Image](https://inspireface-1259028827.cos.ap-singapore.myqcloud.com/docs/cv/rotated.jpg)

---

- **翻转**
    - 支持水平、垂直和双向翻转

```cpp
inspirecv::Image flipped_vertical = img.FlipVertical();
```



![Flipped Image](https://inspireface-1259028827.cos.ap-singapore.myqcloud.com/docs/cv/flipped_vertical.jpg)

---

```cpp
inspirecv::Image flipped_horizontal = img.FlipHorizontal();
```



![Flipped Image](https://inspireface-1259028827.cos.ap-singapore.myqcloud.com/docs/cv/flipped_horizontal.jpg)

---

- **裁剪**

```cpp
inspirecv::Rect<int> rect = inspirecv::Rect<int>::Create(78, 41, 171, 171);
inspirecv::Image cropped = img.Crop(rect);
```



![Cropped Image](https://inspireface-1259028827.cos.ap-singapore.myqcloud.com/docs/cv/cropped.jpg)

---

- **填充**

```cpp
int top = 50, bottom = 50, left = 50, right = 50;
inspirecv::Image padded = img.Pad(top, bottom, left, right, inspirecv::Color::Black);
```



![Padded Image](https://inspireface-1259028827.cos.ap-singapore.myqcloud.com/docs/cv/padded.jpg)

---

- **交换红蓝通道**

```cpp
inspirecv::Image swapped = img.SwapRB();
```



![Swapped Image](https://inspireface-1259028827.cos.ap-singapore.myqcloud.com/docs/cv/swapped.jpg)

---

- **乘法**

```cpp
double scale_factor = 0.5;
inspirecv::Image scaled = img.Mul(scale_factor);
```



![Scaled Image](https://inspireface-1259028827.cos.ap-singapore.myqcloud.com/docs/cv/scaled.jpg)

---

- **加法**

```cpp
double value = -175;
inspirecv::Image added = img.Add(value);
```



![Added Image](https://inspireface-1259028827.cos.ap-singapore.myqcloud.com/docs/cv/added.jpg)

---

- **仿射变换**
    - 类似于 OpenCV 中的 warpAffine

原始输入是旋转 90 度的图像，变换矩阵来自人脸位置：

![Rotated Image](https://inspireface-1259028827.cos.ap-singapore.myqcloud.com/docs/cv/rotated.jpg)

```cpp
/**
 * 从以下矩阵创建变换矩阵
 * [[a11, a12, tx],
 *  [a21, a22, ty]]
 *
 * 人脸裁剪变换矩阵
 * [[0.0, -1.37626, 261.127],
 *  [1.37626, 0.0, 85.1831]]
*/
float a11 = 0.0f;
float a12 = -1.37626f;
float a21 = 1.37626f;
float a22 = 0.0f;
float b1 = 261.127f;
float b2 = 85.1831f;

// 创建变换矩阵：人脸位置变换矩阵
inspirecv::TransformMatrix trans = inspirecv::TransformMatrix::Create(a11, a12, b1, a21, a22, b2);

// dst_width 和 dst_height 是输出图像的大小
int dst_width = 112;
int dst_height = 112;

// 应用仿射变换
inspirecv::Image affine = rotated_90.WarpAffine(trans, dst_width, dst_height);
```



![Affine Image](https://inspireface-1259028827.cos.ap-singapore.myqcloud.com/docs/cv/affine.jpg)

---

## 图像绘制

图像绘制是 InspireCV 的核心功能。它提供了一组在图像上绘制的函数。

- **绘制矩形**

```cpp
inspirecv::Rect<int> new_rect = rect.Square(1.1f);  // 正方形并扩展矩形
int thickness = 3;
draw_img.DrawRect(new_rect, inspirecv::Color::Green, thickness);
```



![Draw Rectangle](https://inspireface-1259028827.cos.ap-singapore.myqcloud.com/docs/cv/draw_rect.jpg)

---

- **绘制圆形**

```cpp
std::vector<inspirecv::Point<int>> points = new_rect.As<int>().ToFourVertices();
for (auto& point : points) {
    draw_img.DrawCircle(point, 1, inspirecv::Color::Red, 5);
}
```



![Draw Circle](https://inspireface-1259028827.cos.ap-singapore.myqcloud.com/docs/cv/draw_circle.jpg)

---

- **绘制线条**

```cpp
draw_img.DrawLine(points[0], points[1], inspirecv::Color::Cyan, 2);
draw_img.DrawLine(points[1], points[2], inspirecv::Color::Magenta, 2);
draw_img.DrawLine(points[2], points[3], inspirecv::Color::Pink, 2);
draw_img.DrawLine(points[3], points[0], inspirecv::Color::Yellow, 2);
```



![Draw Lines](https://inspireface-1259028827.cos.ap-singapore.myqcloud.com/docs/cv/draw_line.jpg)

---

- **填充**

```cpp
draw_img.Fill(new_rect, inspirecv::Color::Purple);
```



![Fill](https://inspireface-1259028827.cos.ap-singapore.myqcloud.com/docs/cv/fill_rect.jpg)

---

- **重置**

```cpp
// 将图像重置为灰色
std::vector<uint8_t> gray_color(img.Width() * img.Height() * 3, 128);
img.Reset(img.Width(), img.Height(), 3, gray_color.data());
```



![Reset](https://inspireface-1259028827.cos.ap-singapore.myqcloud.com/docs/cv/reset.jpg)

## 帧处理

为了简化图像处理，我们设计了一个帧处理器，它包装输入图像，为图像或视频流等帧序列提供灵活支持。它集成了一个处理管道，内置图像解码（**BGR、RGB、BGRA、RGBA、YUV、NV12、NV21**）、旋转、缩放和仿射变换，同时优化内部缓冲以提高性能。

```
**FrameProcess** 是 InspireFace 模块，尚未集成到 InspireCV 库中。
```

### 创建帧处理器

```cpp
// BGR888 作为原始数据
inspirecv::Image raw = inspirecv::Image::Create("test_res/data/bulk/kun_cartoon_crop_r90.jpg", 3);
const uint8_t* buffer = raw.Data();

// 您也可以使用其他图像格式，如 NV21、NV12、RGBA、RGB、BGR、BGRA
const uint8_t* buffer = ...;

// 创建帧处理
auto width = raw.Width();
auto height = raw.Height();
auto rotation_mode = inspirecv::ROTATION_90;
auto data_format = inspirecv::BGR;
inspirecv::FrameProcess frame_process = inspirecv::FrameProcess::Create(buffer, height, width, data_format, rotation_mode);
```



原始数据示例：

![Resized Image](https://inspireface-1259028827.cos.ap-singapore.myqcloud.com/docs/cv/rotated.jpg)

### 管道

- 设置预览大小

```cpp
// 设置预览大小
frame_process.SetPreviewSize(160);

// 或者

// 设置预览缩放
frame_process.SetPreviewScale(0.5f);
```



- **获取变换图像**
    - 将旋转并缩放图像到预览大小

```cpp
inspirecv::Image transform_img = frame_process.GetTransformImage();
```



![Transform Image](https://inspireface-1259028827.cos.ap-singapore.myqcloud.com/docs/cv/transform_img.jpg)

---

- **获取仿射处理图像**

```cpp
/** 
 * 人脸裁剪变换矩阵
 * [[0.0, 0.726607, -61.8946],
 *  [-0.726607, 0.0, 189.737]]
*/

// 人脸裁剪变换矩阵
float a11 = 0.0f;
float a12 = 0.726607f;
float a21 = -0.726607;
float a22 = 0.0f;
float b1 = -61.8946f;
float b2 = 189.737f;
inspirecv::TransformMatrix affine_matrix = inspirecv::TransformMatrix::Create(a11, a12, b1, a21, a22, b2);
int dst_width = 112;
int dst_height = 112;
inspirecv::Image affine_img = frame_process.ExecuteImageAffineProcessing(affine_matrix, dst_width, dst_height);
```



![Affine Processing Image](https://inspireface-1259028827.cos.ap-singapore.myqcloud.com/docs/cv/affine_img.jpg)

## 性能考虑

- 该库使用 Eigen3 进行高效的矩阵运算
- OKCV 后端提供 OpenCV 的轻量级替代方案
- 操作设计为最小化内存分配
- 为并行处理提供线程安全操作

## 线程安全

该库设计为线程安全的。您可以在多线程应用程序中使用它。

## 错误处理

该库使用错误代码和异常来处理错误条件：

- 图像加载/保存错误
- 无效参数
- 内存分配失败
- 后端特定错误

可以使用标准的 try-catch 块捕获错误：

```cpp
try {
    Image img = Image::Create("nonexistent.jpg");
} catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
}
```

