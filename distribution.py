import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from collections import defaultdict

def parse_segmentation_classes(segmentation_dir, color_to_class):
    """
    解析分割类别目录，并统计每个类别的像素数量。

    Args:
        segmentation_dir (str): 分割类别目录路径。
        color_to_class (dict): 颜色到类别的映射字典，形如{颜色: 类别名}。

    Returns:
        dict: 类别到像素数量的映射字典，形如{类别名: 像素数量}。

    """
    class_count = defaultdict(int)

    for seg_file in os.listdir(segmentation_dir):
        if seg_file.endswith('.png'):
            file_path = os.path.join(segmentation_dir, seg_file)
            seg_image = Image.open(file_path).convert('RGB')  # 确保图像是RGB模式
            seg_image_np = np.array(seg_image)
            for color, class_name in color_to_class.items():
                mask = np.all(seg_image_np == np.array(color).reshape(1, 1, 3), axis=-1)
                class_count[class_name] += np.sum(mask)
                
    return class_count

def plot_class_distribution(class_count):
    """
    绘制VOC数据集分割类别分布柱状图

    Args:
        class_count (dict): 类别及其对应的像素数量，格式为 {类别: 像素数量}

    Returns:
        None

    """
    classes = list(class_count.keys())
    counts = list(class_count.values())

    plt.figure(figsize=(10, 5))
    plt.bar(classes, counts, color='skyblue')
    plt.xlabel('Classes')
    plt.ylabel('Number of Pixels')
    plt.title('VOC Dataset Segmentation Class Distribution')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def print_class_distribution(class_count):
    print("Class Distribution:")
    for cls, count in sorted(class_count.items(), key=lambda item: item[1], reverse=True):
        print(f"Class {cls}: {count} pixels")

if __name__ == '__main__':
    # 替换为你的VOC数据集的SegmentationClass目录路径
    segmentation_dir = 'E:\python\mmsegCode\data\VOCCV_dataset\SegmentationClass_5'
    # 定义颜色到类别名称的映射
    # color_to_class = {
    #     (0, 0, 0): 'wall',     # 颜色 (0, 0, 0) 对应 类别 'Background'
    #     (128, 0, 0): 'waste',     # 颜色 (128, 0, 0) 对应 类别 'Class 1'
    #     (0, 128, 0): 'floor',     # 颜色 (0, 128, 0) 对应 类别 'Class 2'
    #     (128, 128, 0): 'floor_sign',   # 颜色 (128, 128, 0) 对应 类别 'Class 3'
    #     (224,224,192): 'background',   # 颜色 (128, 0, 128) 对应 类别 'Class 5'
    # }
    # 定义颜色到类别名称的映射
    color_to_class = {
        (0, 0, 0): 'wall',     # 颜色 (0, 0, 0) 对应 类别 'Background'
        (128, 0, 0): 'step',     # 颜色 (128, 0, 0) 对应 类别 'Class 1'
        (0, 128, 0): 'waste',     # 颜色 (0, 128, 0) 对应 类别 'Class 2'
        (128, 128, 0): 'floor',   # 颜色 (128, 128, 0) 对应 类别 'Class 3'
        (0, 0, 128): 'floor_sign',     # 颜色 (0, 0, 128) 对应 类别 'Class 4'
        # (128, 0, 128): 'glass',     # 颜色 (0, 0, 128) 对应 类别 'Class 4'
        (224,224,192): 'background',   # 颜色 (128, 0, 128) 对应 类别 'Class 5'
    }

    class_count = parse_segmentation_classes(segmentation_dir, color_to_class)
    print_class_distribution(class_count)
    plot_class_distribution(class_count)

