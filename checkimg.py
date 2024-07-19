from PIL import Image
import numpy as np

# 读取图片
image_path = 'E:\python\mmsegCode\data\VOCCV_0704/label_color.jpg'  # 替换为你的图片路径
image = Image.open(image_path)

# 将图片转换为numpy数组
image_np = np.array(image)

# 查看图片的尺寸
print("Image size:", image.size)
print("Image mode:", image.mode)

# 查看某个像素的值（例如左上角像素）
x, y = 0, 0  # 替换为你想查看的像素位置
print(f"Pixel value at ({x}, {y}):", image_np[y, x])

# 打印前10个像素值
print("First 10 pixel values:", image_np[:10, :10])

# 如果图片是RGB模式，可以分别查看R、G、B通道的值
if image.mode == 'RGB':
    r, g, b = image_np[:,:,0], image_np[:,:,1], image_np[:,:,2]
    print("Red channel pixel values:", r)
    print("Green channel pixel values:", g)
    print("Blue channel pixel values:", b)
