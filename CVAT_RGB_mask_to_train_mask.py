"""
用于将 CVAT 导出的 voc segmentation mask 转换成 训练用的 P 模式的灰度图

！！请确保labelmap中每个类别对应的颜色都是不相同的

输入：
1. mask 文件夹
2. 类别融合 txt （包含 background）
    !!! 请确保满足格式要求：每一行类别之后有英文冒号(:)，如果是主类别后面不能有空格

    txt 文件示例：

        background:unknown,waste
        board:
        ceiling:wall

    其中：
    : 之前表示保留的类别
    : 之后表示需要被融合类别，类别以 , 隔开
    background 需要在第一行，转换时可以选择保留或者转成255

requirements:
    最好新建一个虚拟环境
    pip install opencv-python tqdm

用法：
    python CVAT_RGB_mask_to_train_mask.py -i path/to/segmentation_mask -l path/to/label_merge.txt -b False

结果：
    path/to/mask 文件夹下生成转换好的SegmentationClass_Index 文件夹和 result_label.txt
"""

from collections import namedtuple
from PIL import Image
import os
import numpy as np
import cv2
from tqdm import tqdm
import argparse


# 来自 cityscape 数据集
Label = namedtuple('Label', [
    'name',  # The identifier of this label, e.g. 'car', 'person', ... .
    # We use them to uniquely name a class

    'id',  # An integer ID that is associated with this label.

    'trainId',  # Feel free to modify these IDs as suitable for your method.
    # For trainIds, multiple labels might have the same ID. Then, these labels
    # are mapped to the same class in the ground truth images. For the inverse
    # mapping, we use the label that is defined first in the list below.
    # For example, mapping all void-type classes to the same ID in training,
    # might make sense for some approaches.
    # Max value is 255!

    'color',  # The color of this label
])

Label_TrainId = namedtuple('Label_TrainId', [
    'name',
    'trainId',
])


def read_print_labelmap(file_path):
    """
    :param file_path: path of labelmap.txt
    :return: Label format label
    """
    labels = []
    color_values = []
    # 读取 labelmap 中保存的类别和对应的RGB值
    f = open(file_path, "r")
    print('original labels:')
    print('--' * 40)
    for i, line in enumerate(f.readlines()):
        # 略过第一行
        if i == 0:
            continue
        split = line.split(':')
        label_name = split[0]
        label_id = i-1
        label_trainId = label_id
        color_string = split[1]
        color_split = color_string.split(',')
        color_r = int(color_split[0])
        color_g = int(color_split[1])
        color_b = int(color_split[2])
        color_add = color_r*255*255 + color_g*255 + color_b
        assert color_add not in color_values, 'RGB color of label {} is not unique!'.format(label_name)
        print(color_add)
        color_values.append(color_add)
        ith_label = Label(label_name, label_id, label_trainId, (color_r, color_g, color_b))
        # print(ith_label, ',')
        print(label_name + ':')
        labels.append(ith_label)
    f.close()
    print('--' * 40)
    return labels


def read_merge_label(file_path, background_keep):
    """
    读取类别融合文件
    :param file_path:
    :param background_keep: 是否保留background类别
    :return: 1. merged_labels: A list of Label_TrainId (包含所有labels和对应的trainId)
            2. main_labels: A list of Label_TrainId （包含主labels和对应的trainId）
    """

    merged_labels = []
    main_labels = []

    f = open(file_path, "r")
    if not background_keep:
        background_id = 255
        train_id = 0
    else:
        background_id = 0
        train_id = 1
    for idx, line in enumerate(f.readlines()):
        text = line.strip('\n')
        split = text.split(':')
        # 得到主要的类别
        main_label = split[0]
        # 如果类别是背景，trainId 设置成 255
        if main_label == 'background':
            assert idx == 0, 'The \'backgound\' class should be in the first line!'
            merged_labels.append(Label_TrainId(name=main_label, trainId=background_id))
            main_labels.append(Label_TrainId(name=main_label, trainId=background_id))
            if len(split[-1]) > 0:
                sub_labels = split[-1].split(',')
                for sub_label in sub_labels:
                    merged_labels.append(Label_TrainId(name=sub_label, trainId=background_id))

        else:
            merged_labels.append(Label_TrainId(name=main_label, trainId=train_id))
            main_labels.append(Label_TrainId(name=main_label, trainId=train_id))
            # 判断是否有融合的类别
            # print(len(split[-1]))
            if len(split[-1]) > 0:
                sub_labels = split[-1].split(',')
                for sub_label in sub_labels:
                    merged_labels.append(Label_TrainId(name=sub_label, trainId=train_id))
            train_id = train_id + 1
    f.close()
    return merged_labels, main_labels


def change_trainId(labels, merged_labels):
    """
    根据 merge label 文件 修改 label 的 trainId
    :param labels: 从 labelmap 读取的 labels
    :param merged_labels: 从 merge_label 读取的 labels
    :return: 修改 trainId 之后的 labels
    """
    
    print('The num of labels from labelmap: ', len(labels))
    print('The num of labels from label merge txt: ', len(merged_labels))
    print('--' * 40)
    
    if len(labels) != len(merged_labels):
        set1 = set([l.name for l in merged_labels])
        set2 = set([l.name for l in labels])
        different_elements = set1.symmetric_difference(set2)  
        print('diffent labels: ', list(different_elements))
    
    assert len(labels) == len(merged_labels), 'Please check label merge txt'

    result_labels = []
    for label in labels:
        name = label.name
        id = label.id
        color = label.color
        found_flag = False
        # 查找是否有对应的标签
        for merged_label in merged_labels:
            merged_label_name = merged_label.name
            if name == merged_label_name:
                found_flag = True
                train_id = merged_label.trainId
        assert found_flag, "Label '{}' not found in merged labels, please check.".format(label.name)
        result_labels.append(Label(name=name,
                                   id=id,
                                   trainId=train_id,
                                   color=color
                                   ),
                             )
    return result_labels


def label_colormap(N=256):
    """
    生成自定义colormap
    :param N:
    :return: np array of size (N, 3)
    """
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    cmap = np.zeros((N, 3))
    for i in range(0, N):
        id = i
        r, g, b = 0, 0, 0
        for j in range(0, 8):
            r = np.bitwise_or(r, (bitget(id, 0) << 7 - j))
            g = np.bitwise_or(g, (bitget(id, 1) << 7 - j))
            b = np.bitwise_or(b, (bitget(id, 2) << 7 - j))
            id = (id >> 3)
        cmap[i, 0] = r
        cmap[i, 1] = g
        cmap[i, 2] = b
    cmap = cmap.astype(np.float32)
    return cmap

def unpaint_mask(painted_mask, inverse_colormap):
    # Convert color mask to index mask

    # mask: HWC BGR [0; 255]
    # colormap: (R, G, B) -> index
    assert len(painted_mask.shape) == 3

    if callable(inverse_colormap):
        map_fn = lambda a: inverse_colormap(
                (a >> 16) & 255, (a >> 8) & 255, a & 255
            )
    else:
        map_fn = lambda a: inverse_colormap[(
                (a >> 16) & 255, (a >> 8) & 255, a & 255
            )]

    painted_mask = painted_mask.astype(int)
    painted_mask = painted_mask[:, :, 0] + \
                   (painted_mask[:, :, 1] << 8) + \
                   (painted_mask[:, :, 2] << 16)
    uvals, unpainted_mask = np.unique(painted_mask, return_inverse=True)
    # print(uvals)
    palette = np.array([map_fn(v) for v in uvals],
        dtype=np.min_scalar_type(len(uvals)))
    # print(palette)
    # print(unpainted_mask)
    unpainted_mask = palette[unpainted_mask].reshape(painted_mask.shape[:2])

    return unpainted_mask

def rgb2graymask(rgb_dir, gray_dir, unpaint_colormap, save_colormap):
    """
    :param rgb_dir: RGB mask 文件夹
    :param gray_dir: Gray mask 文件夹
    :param unpaint_colormap: RGB 与 Index 的对应关系
    :param save_colormap: 保存 P 模式参考的 palette
    :return:
    """
    rgb_path_list = os.listdir(rgb_dir)
    if not os.path.isdir(gray_dir):
        os.makedirs(gray_dir)

    total = len(rgb_path_list)
    print("total rgb masks: ", total)

    pbar = tqdm(total=total)
    for name in rgb_path_list:
        # print(name)
        mask = cv2.imread(os.path.join(rgb_dir, name))
        # RGB mask (HWC) -> [0; max_index] mask
        u_mask = unpaint_mask(mask, unpaint_colormap)
        im = Image.fromarray(u_mask)
        # 重新保存为P模式的灰度图
        im.putpalette(save_colormap.astype(np.uint8).flatten())
        im.save(os.path.join(gray_dir, name))
        pbar.update(1)
        # if index % 10 == 0:
        #     print("already transformed: ", index)


def write_new_label(new_file_path, labels):
    """
    :param new_file_path: 新的 label txt 文件地址
    :param labels: 修改标签后的label
    :param background_id: background trainId
    :return: new_id: 新的融合后的label
    """
    new_label_file = open(new_file_path, 'w')
    for label in labels:
        if label.trainId == 255:
            continue
        else:
            new_label_file.writelines([label.name, '\n'])

    new_label_file.close()


def draw_label_palette(labels, colormap, file_path):
    """
    :param new_label:
    :param colormap:
    :return:
    """

    img = np.zeros((22 * len(labels), 250, 3), np.uint8)
    xmin = 1
    ymin = 1
    recth = 20
    rectw = 60
    font = cv2.FONT_HERSHEY_SIMPLEX  # 定义字体

    for idx, label in enumerate(labels):
        a = (xmin, ymin + idx * recth)  # 左上角坐标
        b = (xmin + rectw, ymin + idx * recth + recth)  # 右下角坐标
        color = colormap[label.trainId]
        rr = int(color[0])
        gg = int(color[1])
        bb = int(color[2])
        cv2.rectangle(img, a, b, [bb, gg, rr], thickness=-1)
        # 图像，文字内容，坐标(右上角坐标) ，字体，大小，颜色，字体厚度
        cv2.putText(img, label.name, (a[0] + rectw, a[1] + 15), font, 0.6, (255, 255, 255), 1)

    cv2.imwrite(file_path, img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create Train Label Img')
    parser.add_argument('input_folder', type=str,
                        help='folder path of CVAT output segmentation mask 1.1')
    parser.add_argument('label_merge_txt', type=str, default='labels/17.txt',
                        help='label merge txt')
    parser.add_argument('-b', '--keep_background', type=bool, default=False,
                        help=' Ture or Flase. whether Background is set as a class, default is False')

    args = parser.parse_args()
    # ---------------------------------------------------------------
    # 1. （输入） segmentation mask 1.1 文件夹
    mask_dir = args.input_folder
    # 2. （输入）类别融合txt
    label_merge_txt = args.label_merge_txt
    # ---------------------------------------------------------------


    # 找到rgb mask 文件夹
    rgb_mask_dir = os.path.join(mask_dir, "SegmentationClass")
    assert os.path.exists(rgb_mask_dir), "rgb mask folder not found!"

    # 创建灰度 mask 文件夹
    gray_mask_dir = os.path.join(mask_dir, "SegmentationClass_Index")

    # 找到 labelmap 文件，一般位于RGB mask文件夹同一级
    labelmap_file = os.path.join(mask_dir, "labelmap.txt")
    assert os.path.exists(labelmap_file), "labelmap not found!"

    # 读取 labelmap 文件
    labels = read_print_labelmap(labelmap_file)

    # 读取类别融合文件
    # merged_labels: 包含所有的类别和对应的trainId
    # main_labels: 主类别，用于生成result_label.txt
    merged_labels, main_labels = read_merge_label(label_merge_txt, args.keep_background)

    # 修改labelmap读取的label的trainId
    labels = change_trainId(labels, merged_labels)
    print("-"*20)
    for la in labels:
        print(la)
    print('-'*20)

    # 获得 RGB 和 [0; max_index] 的对应关系
    unpaint_colormap = {label.color: label.trainId for label in labels}

    # 创建P模式的palette
    save_colormap = label_colormap(256)

    new_label_file_path = os.path.join(mask_dir, "result_label.txt")
    # 生成标签融合后的label文件
    write_new_label(new_label_file_path, main_labels)

    # print(merged_labels)
    label_color_path = os.path.join(mask_dir, "label_color.jpg")
    # 创建类别和颜色对照图
    draw_label_palette(main_labels, save_colormap, label_color_path)

    # 将RGB mask 转化成 灰度 mask，并保存为P模式
    rgb2graymask(rgb_mask_dir, gray_mask_dir, unpaint_colormap, save_colormap)
    
    print('---'*20)
    print('Result SegmentationClass_Index saved in: ', gray_mask_dir)
    print('result label saved: ', new_label_file_path)
    print('The color of the labels saved: ', label_color_path)





