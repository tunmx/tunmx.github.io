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


## 基础图像处理接口设计

在基础图像处理中，最常见的莫过于几种常见的图像变换函数比如：

1. 平移(Translation)
2. 缩放(Scaling)
3. 旋转(Rotation)
4. 切变(Shear)
5. 裁剪(Crop)
6. 镜像(Mirror)
7. 翻转(Flip)
8. 旋转(Rotate)
9. **仿射变换(Affine)**
10. 模糊(Blur)
11. 补边(Padding)
12. 填充(Fill)
13. 绘制(Draw)
14. ....待补充

### Image类设计

所以基于以上几种常见的图像变换函数，我们设计以下几个基础的图像处理接口：

```cpp
namespace inspirecv {

/**
 * @brief Class representing an image with basic image processing operations
 */
class INSPIRECV_API Image {
public:
    // Disable copy constructor and copy assignment operator
    Image(const Image&) = delete;
    Image& operator=(const Image&) = delete;

    // #endif  // INSPIRECV_BACKEND_OPENCV

    /**
     * @brief Default constructor
     */
    Image();

    /**
     * @brief Destructor
     */
    ~Image();

    /**
     * @brief Move constructor
     */
    Image(Image&&) noexcept;

    /**
     * @brief Move assignment operator
     */
    Image& operator=(Image&&) noexcept;

    /**
     * @brief Constructor with width, height, channels and optional data
     * @param width Image width
     * @param height Image height
     * @param channels Number of color channels
     * @param copy_data Whether to copy data, default is true. When set to false, the image will
     * directly use the input data without copying. This is dangerous as the data lifetime must be
     * managed externally and outlive the image. The data must also remain valid and unmodified for
     * the entire lifetime of the image.
     * Note: Zero-copy mode is currently only supported with OKCV backend.
     * @param data Optional pointer to image data
     */
    Image(int width, int height, int channels, const uint8_t* data = nullptr,
          bool copy_data = true);

    /**
     * @brief Reset image with new dimensions and data
     * @param width New width
     * @param height New height
     * @param channels New number of channels
     * @param data Optional pointer to new image data
     */
    void Reset(int width, int height, int channels, const uint8_t* data = nullptr);

    /**
     * @brief Create a deep copy of the image
     * @return New image that is a copy of this one
     */
    Image Clone() const;

    /**
     * @brief Get image width
     * @return Width in pixels
     */
    int Width() const;

    /**
     * @brief Get image height
     * @return Height in pixels
     */
    int Height() const;

    /**
     * @brief Get number of color channels
     * @return Number of channels
     */
    int Channels() const;

    /**
     * @brief Check if image is empty
     * @return true if image has no data
     */
    bool Empty() const;

    /**
     * @brief Get raw image data
     * @return Pointer to image data
     */
    const uint8_t* Data() const;

    /**
     * @brief Get internal image implementation
     * @return Pointer to internal image
     */
    void* GetInternalImage() const;

    /**
     * @brief Read image from file
     * @param filename Path to image file
     * @param channels Number of channels to read (default: 3)
     * @return true if successful
     */
    bool Read(const std::string& filename, int channels = 3);

    /**
     * @brief Write image to file
     * @param filename Output file path
     * @return true if successful
     */
    bool Write(const std::string& filename) const;

    /**
     * @brief Display image in a window
     * @param window_name Name of display window
     * @param delay Wait time in milliseconds (0 = wait forever)
     */
    void Show(const std::string& window_name = std::string("win"), int delay = 0) const;

    /**
     * @brief Fill entire image with value
     * @param value Fill value
     */
    void Fill(double value);

    /**
     * @brief Multiply image by scale factor
     * @param scale Scale factor
     * @return New scaled image
     */
    Image Mul(double scale) const;

    /**
     * @brief Add value to image
     * @param value Value to add
     * @return New image with added value
     */
    Image Add(double value) const;

    /**
     * @brief Resize image to new dimensions
     * @param width New width
     * @param height New height
     * @param use_linear Use linear interpolation if true
     * @return Resized image
     */
    Image Resize(int width, int height, bool use_linear = true) const;

    /**
     * @brief Crop image to rectangle
     * @param rect Rectangle defining crop region
     * @return Cropped image
     */
    Image Crop(const Rect<int>& rect) const;

    /**
     * @brief Apply affine transformation
     * @param matrix 2x3 transformation matrix
     * @param width Output width
     * @param height Output height
     * @return Transformed image
     */
    Image WarpAffine(const TransformMatrix& matrix, int width, int height) const;

    /**
     * @brief Rotate image 90 degrees clockwise
     * @return Rotated image
     */
    Image Rotate90() const;

    /**
     * @brief Rotate image 180 degrees
     * @return Rotated image
     */
    Image Rotate180() const;

    /**
     * @brief Rotate image 270 degrees clockwise
     * @return Rotated image
     */
    Image Rotate270() const;

    /**
     * @brief Swap the red and blue channels of the image.
     * @return The swapped image.
     */
    Image SwapRB() const;

    /**
     * @brief Flip image horizontally
     * @return Flipped image
     */
    Image FlipHorizontal() const;

    /**
     * @brief Flip image vertically
     * @return Flipped image
     */
    Image FlipVertical() const;

    /**
     * @brief Add padding around image
     * @param top Top padding
     * @param bottom Bottom padding
     * @param left Left padding
     * @param right Right padding
     * @param color Padding color values
     * @return Padded image
     */
    Image Pad(int top, int bottom, int left, int right, const std::vector<double>& color) const;

    /**
     * @brief Apply Gaussian blur
     * @param kernel_size Size of Gaussian kernel
     * @param sigma Gaussian standard deviation
     * @return Blurred image
     */
    Image GaussianBlur(int kernel_size, double sigma) const;

    /**
     * @brief Apply threshold operation
     * @param thresh Threshold value
     * @param maxval Maximum value
     * @param type Threshold type
     * @return Thresholded image
     */
    Image Threshold(double thresh, double maxval, int type) const;

    /**
     * @brief Convert image to grayscale
     * @return Grayscale image
     */
    Image ToGray() const;

    /**
     * @brief Draw line between two points
     * @param p1 Start point
     * @param p2 End point
     * @param color Line color
     * @param thickness Line thickness
     */
    void DrawLine(const Point<int>& p1, const Point<int>& p2, const std::vector<double>& color,
                  int thickness = 1);

    /**
     * @brief Draw rectangle
     * @param rect Rectangle to draw
     * @param color Rectangle color
     * @param thickness Line thickness
     */
    void DrawRect(const Rect<int>& rect, const std::vector<double>& color, int thickness = 1);

    /**
     * @brief Draw circle
     * @param center Circle center point
     * @param radius Circle radius
     * @param color Circle color
     * @param thickness Line thickness
     */
    void DrawCircle(const Point<int>& center, int radius, const std::vector<double>& color,
                    int thickness = 1);

    /**
     * @brief Fill rectangle with color
     * @param rect Rectangle to fill
     * @param color Fill color
     */
    void Fill(const Rect<int>& rect, const std::vector<double>& color);

    /**
     * @brief Create image with dimensions and optional data
     * @param width Image width
     * @param height Image height
     * @param channels Number of channels
     * @param data Optional image data
     * @param copy_data Whether to copy data, default is true. When set to false, the image will
     * directly use the input data without copying. This is dangerous as the data lifetime must be
     * managed externally and outlive the image. The data must also remain valid and unmodified for
     * the entire lifetime of the image.
     * Note: Zero-copy mode is currently only supported with OKCV backend.
     * @return New image
     */
    static Image Create(int width, int height, int channels, const uint8_t* data = nullptr,
                        bool copy_data = true);

    /**
     * @brief Create empty image
     * @return New empty image
     */
    static Image Create();

    /**
     * @brief Create image from file
     * @param filename Path to image file
     * @param channels Number of channels to read
     * @return New image loaded from file
     */
    static Image Create(const std::string& filename, int channels = 3);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;

    /**
     * @brief Private constructor with implementation
     * @param impl Image implementation
     */
    Image(std::unique_ptr<Impl> impl);

    // Declare Impl as friend class
    friend class Impl;
    // Add this line to declare the operator as a friend
    friend std::ostream& operator<<(std::ostream& os, const Image& image);
};

/**
 * @brief Stream output operator for Image
 * @param os Output stream
 * @param image Image to output
 * @return Modified output stream
 */
INSPIRECV_API std::ostream& operator<<(std::ostream& os, const Image& image);

}  // namespace inspirecv
```

### 基础几何类设计

当然，除了基础的图像处理接口，我们还需要设计一些基础的几何类，比如：

1. 点(Point)
2. 矩形框(Rect)
3. 颜色(Color)
4. 尺寸(Size)
5. 变换矩阵(TransformMatrix)

#### 变换矩阵类设计

```cpp
namespace inspirecv {

/**
 * @brief Class representing a 2D transformation matrix
 */
class INSPIRECV_API TransformMatrix {
public:
    /**
     * @brief Copy constructor
     * @param other Matrix to copy from
     */
    TransformMatrix(const TransformMatrix &other);

    /**
     * @brief Copy assignment operator
     * @param other Matrix to copy from
     * @return Reference to this matrix
     */
    TransformMatrix &operator=(const TransformMatrix &other);

    /**
     * @brief Default constructor
     */
    TransformMatrix();

    /**
     * @brief Destructor
     */
    ~TransformMatrix();

    /**
     * @brief Constructor with matrix elements
     * @param a11 Element at row 1, col 1
     * @param a12 Element at row 1, col 2
     * @param b1 Element at row 1, col 3
     * @param a21 Element at row 2, col 1
     * @param a22 Element at row 2, col 2
     * @param b2 Element at row 2, col 3
     */
    TransformMatrix(float a11, float a12, float b1, float a21, float a22, float b2);

    // Basic getters and setters
    /**
     * @brief Get matrix element at specified position
     * @param row Row index (0-based)
     * @param col Column index (0-based)
     * @return Value at specified position
     */
    float Get(int row, int col) const;

    /**
     * @brief Set matrix element at specified position
     * @param row Row index (0-based)
     * @param col Column index (0-based)
     * @param value New value to set
     */
    void Set(int row, int col, float value);

    /**
     * @brief Convert matrix to vector form
     * @return Vector containing matrix elements in row-major order
     */
    std::vector<float> Squeeze() const;

    /**
     * @brief Array subscript operator for const access
     * @param index Element index in row-major order
     * @return Value at specified index
     */
    float operator[](int index) const;

    /**
     * @brief Array subscript operator for modifying access
     * @param index Element index in row-major order
     * @return Reference to value at specified index
     */
    float &operator[](int index);

    /**
     * @brief Get internal matrix implementation
     * @return Pointer to internal matrix implementation
     */
    void *GetInternalMatrix() const;

    // Basic operations
    /**
     * @brief Check if matrix is identity matrix
     * @return true if matrix is identity
     */
    bool IsIdentity() const;

    /**
     * @brief Set matrix to identity matrix
     */
    void SetIdentity();

    /**
     * @brief Invert this matrix in-place
     */
    void Invert();

    /**
     * @brief Get inverse of this matrix
     * @return New matrix that is inverse of this one
     */
    TransformMatrix GetInverse() const;

    // Transform operations
    /**
     * @brief Apply translation transformation
     * @param dx Translation in x direction
     * @param dy Translation in y direction
     */
    void Translate(float dx, float dy);

    /**
     * @brief Apply scaling transformation
     * @param sx Scale factor in x direction
     * @param sy Scale factor in y direction
     */
    void Scale(float sx, float sy);

    /**
     * @brief Apply rotation transformation
     * @param angle Rotation angle in radians
     */
    void Rotate(float angle);

    // Matrix operations
    /**
     * @brief Multiply this matrix with another
     * @param other Matrix to multiply with
     * @return Result of matrix multiplication
     */
    TransformMatrix Multiply(const TransformMatrix &other) const;

    /**
     * @brief Create a deep copy of this matrix
     * @return New matrix that is a copy of this one
     */
    TransformMatrix Clone() const;

    // Factory methods
    /**
     * @brief Create default transformation matrix
     * @return New default matrix
     */
    static TransformMatrix Create();

    /**
     * @brief Create matrix with specified elements
     * @param a11 Element at row 1, col 1
     * @param a12 Element at row 1, col 2
     * @param b1 Element at row 1, col 3
     * @param a21 Element at row 2, col 1
     * @param a22 Element at row 2, col 2
     * @param b2 Element at row 2, col 3
     * @return New matrix with specified elements
     */
    static TransformMatrix Create(float a11, float a12, float b1, float a21, float a22, float b2);

    /**
     * @brief Create identity matrix
     * @return New identity matrix
     */
    static TransformMatrix Identity();

    /**
     * @brief Create 90-degree rotation matrix
     * @return New matrix for 90-degree rotation
     */
    static TransformMatrix Rotate90();

    /**
     * @brief Create 180-degree rotation matrix
     * @return New matrix for 180-degree rotation
     */
    static TransformMatrix Rotate180();

    /**
     * @brief Create 270-degree rotation matrix
     * @return New matrix for 270-degree rotation
     */
    static TransformMatrix Rotate270();

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

/**
 * @brief Stream output operator for TransformMatrix
 * @param os Output stream
 * @param matrix Matrix to output
 * @return Reference to output stream
 */
INSPIRECV_API std::ostream &operator<<(std::ostream &os, const TransformMatrix &matrix);

}  // namespace inspirecv
```


#### 点类设计

```cpp
namespace inspirecv {


 * @brief A template class representing a 2D point with x and y coordinates
 * @tparam T The coordinate type (int, float, double)
 */
template <typename T>
class INSPIRECV_API Point {
public:
    /**
     * @brief Default constructor
     */
    Point();

    /**
     * @brief Constructor with x and y coordinates
     * @param x The x coordinate
     * @param y The y coordinate
     */
    Point(T x, T y);

    /**
     * @brief Destructor
     */
    ~Point();

    // Move semantics
    /**
     * @brief Move constructor
     */
    Point(Point &&) noexcept;

    /**
     * @brief Move assignment operator
     */
    Point &operator=(Point &&) noexcept;

    // Disable copy
    /**
     * @brief Copy constructor
     */
    Point(const Point &other);

    /**
     * @brief Copy assignment operator
     */
    Point &operator=(const Point &other);

    /**
     * @brief Equality operator
     * @param other Point to compare with
     * @return true if points are equal
     */
    bool operator==(const Point &other) const;

    /**
     * @brief Inequality operator
     * @param other Point to compare with
     * @return true if points are not equal
     */
    bool operator!=(const Point &other) const;

    /**
     * @brief Convert point coordinates to another type
     * @tparam U Target type for conversion
     * @return New point with converted coordinates
     */
    template <typename U>
    Point<U> As() const;

    // Basic getters and setters
    /**
     * @brief Get x coordinate
     * @return The x coordinate
     */
    T GetX() const;

    /**
     * @brief Get y coordinate
     * @return The y coordinate
     */
    T GetY() const;

    /**
     * @brief Set x coordinate
     * @param x New x coordinate value
     */
    void SetX(T x);

    /**
     * @brief Set y coordinate
     * @param y New y coordinate value
     */
    void SetY(T y);

    /**
     * @brief Get internal point implementation
     * @return Pointer to internal point implementation
     */
    void *GetInternalPoint() const;

    // Basic operations
    /**
     * @brief Calculate length (magnitude) of vector from origin to point
     * @return Length of vector
     */
    double Length() const;

    /**
     * @brief Calculate Euclidean distance to another point
     * @param other Point to calculate distance to
     * @return Distance between points
     */
    double Distance(const Point &other) const;

    /**
     * @brief Calculate dot product with another point
     * @param other Point to calculate dot product with
     * @return Dot product result
     */
    T Dot(const Point &other) const;

    /**
     * @brief Calculate cross product with another point
     * @param other Point to calculate cross product with
     * @return Cross product result
     */
    T Cross(const Point &other) const;

    /**
     * @brief Factory method to create a new point
     * @param x The x coordinate
     * @param y The y coordinate
     * @return New point instance
     */
    static Point Create(T x, T y);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};  // class Point

/** @brief Type alias for Point<int> */
using Point2i = Point<int>;
/** @brief Type alias for Point<float> */
using Point2f = Point<float>;
/** @brief Type alias for Point<double> */
using Point2d = Point<double>;
/** @brief Type alias for Point<int> */
using Point2 = Point2i;

/**
 * @brief Stream output operator for Point
 * @param os Output stream
 * @param point Point to output
 * @return Reference to output stream
 */
template <typename T>
INSPIRECV_API std::ostream &operator<<(std::ostream &os, const Point<T> &point);

/**
 * @brief Stream output operator for vector of Points
 * @param os Output stream
 * @param points Vector of points to output
 * @return Reference to output stream
 */
template <typename T>
INSPIRECV_API std::ostream &operator<<(std::ostream &os, const std::vector<Point<T>> &points);

/**
 * @brief Apply transformation matrix to vector of points
 * @param points Vector of points to transform
 * @param transform Transform matrix to apply
 * @return Vector of transformed points
 */
template <typename T>
INSPIRECV_API std::vector<Point<T>> ApplyTransformToPoints(const std::vector<Point<T>> &points,
                                                           const TransformMatrix &transform);

/**
 * @brief Estimate similarity transform between two sets of corresponding points
 * @param src_points Source points
 * @param dst_points Destination points
 * @return Estimated transform matrix
 */
template <typename T>
INSPIRECV_API TransformMatrix SimilarityTransformEstimate(const std::vector<Point<T>> &src_points,
                                                          const std::vector<Point<T>> &dst_points);

/**
 * @brief Estimate similarity transform between two sets of corresponding points using Umeyama
 * algorithm
 * @param src_points Source points
 * @param dst_points Destination points
 * @return Estimated transform matrix
 */
template <typename T>
INSPIRECV_API TransformMatrix SimilarityTransformEstimateUmeyama(
  const std::vector<Point<T>> &src_points, const std::vector<Point<T>> &dst_points);

}  // namespace inspirecv
```

#### 矩形框类设计

```cpp
namespace inspirecv {


/**
 * @brief A template class representing a 2D rectangle with position and size
 * @tparam T The coordinate type (int, float, double)
 */
template <typename T>
class INSPIRECV_API Rect {
public:
    /**
     * @brief Copy constructor
     * @param other Rectangle to copy from
     */
    Rect(const Rect &other);

    /**
     * @brief Copy assignment operator
     * @param other Rectangle to copy from
     * @return Reference to this rectangle
     */
    Rect &operator=(const Rect &other);

    /**
     * @brief Default constructor
     */
    Rect();

    /**
     * @brief Constructor with position and size
     * @param x X coordinate of top-left corner
     * @param y Y coordinate of top-left corner
     * @param width Width of rectangle
     * @param height Height of rectangle
     */
    Rect(T x, T y, T width, T height);

    /**
     * @brief Destructor
     */
    ~Rect();

    // Basic getters and setters
    /**
     * @brief Convert rectangle coordinates to another type
     * @tparam U Target type for conversion
     * @return New rectangle with converted coordinates
     */
    template <typename U>
    Rect<U> As() const;

    /**
     * @brief Get x coordinate of top-left corner
     * @return The x coordinate
     */
    T GetX() const;

    /**
     * @brief Get y coordinate of top-left corner
     * @return The y coordinate
     */
    T GetY() const;

    /**
     * @brief Get width of rectangle
     * @return The width
     */
    T GetWidth() const;

    /**
     * @brief Get height of rectangle
     * @return The height
     */
    T GetHeight() const;

    /**
     * @brief Set x coordinate of top-left corner
     * @param x New x coordinate value
     */
    void SetX(T x);

    /**
     * @brief Set y coordinate of top-left corner
     * @param y New y coordinate value
     */
    void SetY(T y);

    /**
     * @brief Set width of rectangle
     * @param width New width value
     */
    void SetWidth(T width);

    /**
     * @brief Set height of rectangle
     * @param height New height value
     */
    void SetHeight(T height);

    /**
     * @brief Get internal rectangle implementation
     * @return Pointer to internal rectangle implementation
     */
    void *GetInternalRect() const;

    // Boundary points
    /**
     * @brief Get top-left corner point
     * @return Point at top-left corner
     */
    Point<T> TopLeft() const;

    /**
     * @brief Get top-right corner point
     * @return Point at top-right corner
     */
    Point<T> TopRight() const;

    /**
     * @brief Get bottom-left corner point
     * @return Point at bottom-left corner
     */
    Point<T> BottomLeft() const;

    /**
     * @brief Get bottom-right corner point
     * @return Point at bottom-right corner
     */
    Point<T> BottomRight() const;

    /**
     * @brief Get center point of rectangle
     * @return Point at center of rectangle
     */
    Point<T> Center() const;

    /**
     * @brief Convert rectangle to four corner vertices
     * @return Vector of four corner points
     */
    std::vector<Point<T>> ToFourVertices() const;

    /**
     * @brief Get a safe rectangle bounded by given dimensions
     * @param width Maximum width bound
     * @param height Maximum height bound
     * @return New rectangle within bounds
     */
    Rect<T> SafeRect(T width, T height) const;

    // Basic operations
    /**
     * @brief Calculate area of rectangle
     * @return Area value
     */
    T Area() const;

    /**
     * @brief Check if rectangle is empty
     * @return true if width or height is zero
     */
    bool Empty() const;

    /**
     * @brief Check if point is inside rectangle
     * @param point Point to check
     * @return true if point is inside
     */
    bool Contains(const Point<T> &point) const;

    /**
     * @brief Check if another rectangle is fully contained
     * @param rect Rectangle to check
     * @return true if rect is fully inside
     */
    bool Contains(const Rect<T> &rect) const;

    // Geometric operations
    /**
     * @brief Calculate intersection with another rectangle
     * @param other Rectangle to intersect with
     * @return Intersection rectangle
     */
    Rect<T> Intersect(const Rect<T> &other) const;

    /**
     * @brief Calculate union with another rectangle
     * @param other Rectangle to unite with
     * @return Union rectangle
     */
    Rect<T> Union(const Rect<T> &other) const;

    /**
     * @brief Calculate Intersection over Union (IoU)
     * @param other Rectangle to calculate IoU with
     * @return IoU value between 0 and 1
     */
    double IoU(const Rect<T> &other) const;

    // Transformation operations
    /**
     * @brief Scale rectangle dimensions
     * @param sx Scale factor for width
     * @param sy Scale factor for height
     */
    void Scale(T sx, T sy);

    /**
     * @brief Translate rectangle position
     * @param dx Translation in x direction
     * @param dy Translation in y direction
     */
    void Translate(T dx, T dy);

    /**
     * @brief Create a square rectangle centered on current rectangle
     * @param scale Scale factor for square size
     * @return Square rectangle
     */
    Rect<T> Square(float scale = 1.0) const;

    /**
     * @brief Create rectangle from coordinates and dimensions
     * @param x X coordinate of top-left corner
     * @param y Y coordinate of top-left corner
     * @param width Width of rectangle
     * @param height Height of rectangle
     * @return New rectangle
     */
    static Rect<T> Create(T x, T y, T width, T height);

    /**
     * @brief Create rectangle from two corner points
     * @param left_top Top-left corner point
     * @param right_bottom Bottom-right corner point
     * @return New rectangle
     */
    static Rect<T> Create(const Point<T> &left_top, const Point<T> &right_bottom);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

/**
 * @brief Type alias for integer rectangle
 */
using Rect2i = Rect<int>;

/**
 * @brief Type alias for float rectangle
 */
using Rect2f = Rect<float>;

/**
 * @brief Type alias for double rectangle
 */
using Rect2d = Rect<double>;

/** @brief Type alias for Rect<int> */
using Rect2 = Rect2i;

/**
 * @brief Stream output operator for rectangle
 * @param os Output stream
 * @param rect Rectangle to output
 * @return Reference to output stream
 */
template <typename T>
INSPIRECV_API std::ostream &operator<<(std::ostream &os, const Rect<T> &rect);

/**
 * @brief Calculate minimum bounding rectangle for set of points
 * @param points Vector of points
 * @return Minimum bounding rectangle
 */
template <typename T>
INSPIRECV_API Rect<T> MinBoundingRect(const std::vector<Point<T>> &points);

/**
 * @brief Apply transformation matrix to rectangle
 * @param rect Rectangle to transform
 * @param transform Transformation matrix to apply
 * @return Transformed rectangle
 */
template <typename T>
INSPIRECV_API Rect<T> ApplyTransformToRect(const Rect<T> &rect, const TransformMatrix &transform);

}  // namespace inspirecv
```

#### 尺寸类设计

```cpp
namespace inspirecv {


/**
 * @brief A template class representing a 2D size with width and height
 * @tparam T The size type (int, float, double)
 */
template <typename T>
class INSPIRECV_API Size {
public:
    /**
     * @brief Copy constructor
     * @param other Size to copy from
     */
    Size(const Size &other);

    /**
     * @brief Copy assignment operator
     * @param other Size to copy from
     * @return Reference to this size
     */
    Size &operator=(const Size &other);

    /**
     * @brief Default constructor
     */
    Size();

    /**
     * @brief Constructor with width and height
     * @param width Width value
     * @param height Height value
     */
    Size(T width, T height);

    /**
     * @brief Destructor
     */
    ~Size();

    // Basic getters and setters
    /**
     * @brief Get width value
     * @return The width
     */
    T GetWidth() const;

    /**
     * @brief Get height value
     * @return The height
     */
    T GetHeight() const;

    /**
     * @brief Set width value
     * @param width New width value
     */
    void SetWidth(T width);

    /**
     * @brief Set height value
     * @param height New height value
     */
    void SetHeight(T height);

    // Basic operations
    /**
     * @brief Calculate area (width * height)
     * @return Area value
     */
    T Area() const;

    /**
     * @brief Check if size is empty (width or height is 0)
     * @return true if empty
     */
    bool Empty() const;

    /**
     * @brief Scale width and height by given factors
     * @param sx Scale factor for width
     * @param sy Scale factor for height
     */
    void Scale(T sx, T sy);

    // Factory method
    /**
     * @brief Create a new Size instance
     * @param width Width value
     * @param height Height value
     * @return New Size instance
     */
    static Size Create(T width, T height);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

/**
 * @brief Type alias for Size with integer values
 */
using Size2i = Size<int>;

/**
 * @brief Type alias for Size with float values
 */
using Size2f = Size<float>;

/**
 * @brief Type alias for Size with double values
 */
using Size2d = Size<double>;

/** @brief Type alias for Size<int> */
using Size2 = Size2i;

/**
 * @brief Stream output operator for Size
 * @param os Output stream
 * @param size Size to output
 * @return Reference to output stream
 */
template <typename T>
INSPIRECV_API std::ostream &operator<<(std::ostream &os, const Size<T> &size);

}  // namespace inspirecv
```

### 具体实现

有了以上接口定义，具体实现就相对简单了，这里我先实现了OpenCV的接口，再对应去手动实现自定义的接口。值得一提的是，从具体实现的效果来看，无论是在x86还是ARM上（ARM上的Affine变换使用了NEON优化），速度上均接近于OpenCV，鉴于个人编码水平，想说整体超过OpenCV的速度是比较困难的，但是至少处理效果达到与OpenCV一致，并且速度上比OpenCV慢也是只是慢了毫秒级别，所以整体来说，效果和速度达到了可接受的范围内。

具体实现可参考项目地址: [inspirecv](https://github.com/tunmx/inspirecv)

### 硬件相关的集成

通过以上基本的实现，我尝试将该库与Rockchip的RGA加速进行结合集成，让RGA加速库与该库的接口进行适配，从而让该库在Rockchip的平台上也能享受到RGA加速带来的性能提升。

```cpp
namespace nexus {

class RgaImageProcessor : public ImageProcessor {
public:
    RgaImageProcessor();
    ~RgaImageProcessor() override;

    int32_t Resize(const uint8_t* src_data, int src_width, int src_height, int channels, uint8_t** dst_data, int dst_width, int dst_height) override;

    int32_t SwapColor(const uint8_t* src_data, int src_width, int src_height, int channels, uint8_t** dst_data) override;

    int32_t Padding(const uint8_t* src_data, int src_width, int src_height, int channels, int top, int bottom, int left, int right,
                    uint8_t** dst_data, int& dst_width, int& dst_height) override;

    int32_t ResizeAndPadding(const uint8_t* src_data, int src_width, int src_height, int channels, int dst_width, int dst_height, uint8_t** dst_data,
                             float& scale) override;

    int32_t MarkDone() override;

public:
    struct BufferInfo {
        int dma_fd;
        int width;
        int height;
        int channels;
        size_t buffer_size;
    };

    BufferInfo GetCurrentSrcBufferInfo() const {
        auto it = buffer_cache_.find(last_src_key_);
        if (it != buffer_cache_.end()) {
            const auto& buffer = it->second;
            return {buffer.dma_fd, buffer.width, buffer.height, buffer.channels, buffer.buffer_size};
        }
        return {-1, 0, 0, 0, 0};  // Return invalid values if cache doesn't exist
    }

    BufferInfo GetCurrentDstBufferInfo() const {
        auto it = buffer_cache_.find(last_dst_key_);
        if (it != buffer_cache_.end()) {
            const auto& buffer = it->second;
            return {buffer.dma_fd, buffer.width, buffer.height, buffer.channels, buffer.buffer_size};
        }
        return {-1, 0, 0, 0, 0};  // Return invalid values if cache doesn't exist
    }

    size_t GetCacheSize() const {
        return buffer_cache_.size();
    }

    void DumpCacheStatus() const override {
        INSPIRECV_LOG(INFO) << "Current cache status:";
        INSPIRECV_LOG(INFO) << "Cache size: " << buffer_cache_.size();

        auto src_info = GetCurrentSrcBufferInfo();
        INSPIRECV_LOG(INFO) << "Source buffer: "
                            << "dma_fd=" << src_info.dma_fd << ", size=" << src_info.width << "x" << src_info.height << "x" << src_info.channels;

        auto dst_info = GetCurrentDstBufferInfo();
        INSPIRECV_LOG(INFO) << "Destination buffer: "
                            << "dma_fd=" << dst_info.dma_fd << ", size=" << dst_info.width << "x" << dst_info.height << "x" << dst_info.channels;
    }

private:
    struct RGABuffer {
        int width{0};
        int height{0};
        int channels{0};
        int dma_fd{-1};
        void* virtual_addr{nullptr};
        size_t buffer_size{0};
        rga_buffer_handle_t handle{0};
        rga_buffer_t buffer{};

        bool Allocate(int w, int h, int c) {
            width = w;
            height = h;
            channels = c;
            buffer_size = width * height * channels;

            int ret = dma_buf_alloc(INSPIRE_LAUNCH->GetRockchipDmaHeapPath().c_str(), buffer_size, &dma_fd, &virtual_addr);
            if (ret < 0) {
                INSPIRECV_LOG(ERROR) << "Failed to allocate DMA buffer: " << ret;
                return false;
            }

            handle = importbuffer_fd(dma_fd, buffer_size);
            if (handle == 0) {
                INSPIRECV_LOG(ERROR) << "Failed to import buffer";
                Release();
                return false;
            }

            buffer = wrapbuffer_handle(handle, w, h, RK_FORMAT_RGB_888);

            return true;
        }

        void Release() {
            if (handle) {
                releasebuffer_handle(handle);
                handle = 0;
            }
            if (dma_fd >= 0) {
                dma_buf_free(buffer_size, &dma_fd, virtual_addr);
                dma_fd = -1;
                virtual_addr = nullptr;
            }
        }

        ~RGABuffer() {
            Release();
        }
    };

    struct BufferKey {
        int width;
        int height;
        int channels;

        bool operator==(const BufferKey& other) const {
            return width == other.width && height == other.height && channels == other.channels;
        }
    };

    struct BufferKeyHash {
        std::size_t operator()(const BufferKey& key) const {
            return std::hash<int>()(key.width) ^ (std::hash<int>()(key.height) << 1) ^ (std::hash<int>()(key.channels) << 2);
        }
    };

    RGABuffer& GetOrCreateBuffer(const BufferKey& key, bool is_src = true) {
        auto it = buffer_cache_.find(key);
        if (it != buffer_cache_.end()) {
            if (is_src) {
                last_src_key_ = key;
            } else {
                last_dst_key_ = key;
            }
            return it->second;
        }

        if (buffer_cache_.size() >= 3) {  // Keep max 3 buffers in cache
            for (auto it = buffer_cache_.begin(); it != buffer_cache_.end();) {
                if (!(it->first == last_src_key_) && !(it->first == last_dst_key_)) {
                    it = buffer_cache_.erase(it);
                } else {
                    ++it;
                }
            }
        }

        auto& buffer = buffer_cache_[key];
        if (!buffer.Allocate(key.width, key.height, key.channels)) {
            INSPIRECV_LOG(ERROR) << "Failed to allocate RGA buffer";
            throw std::runtime_error("RGA buffer allocation failed");
        }

        if (is_src) {
            last_src_key_ = key;
        } else {
            last_dst_key_ = key;
        }

        return buffer;
    }

private:
    std::unordered_map<BufferKey, RGABuffer, BufferKeyHash> buffer_cache_;
    BufferKey last_src_key_{0, 0, 0};
    BufferKey last_dst_key_{0, 0, 0};
};

}  // namespace nexus

```

### 其他待处理的事件

从最初的计划到目前开发的阶段，发现我的任务就是在手动实现一部分OpenCV的功能和集成一些第三方设备的图像处理加速库，从而适配不同平台和设备上，当然多次想在这个库上添加一些新的功能时，我发现如果继续写下去，这个库会越来越臃肿，最后会干了跟OpenCV和dlib一样的活，并且性能上还不如这些成熟的库，所以我认为这个库到了这部分功能就该适可而止了，起码对于inspireface这个项目而言，这个库的图像处理完全是处于**够用**的状态，后续如果要继续优化可能也只是横向扩展去适配一些第三方厂家提供的图像加速处理后端，而不是深度去优化手动实现的那些在CPU运行的函数。