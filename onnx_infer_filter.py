"""
1. ONNX 推理， ONNX 输出 NCHW
2. 过滤问题图片
3. 保存问题图片

过滤规则：
1. 置信度
2. 边缘重合度
"""

import onnxruntime
from PIL import Image
import numpy as np
import cv2
from pathlib import Path
import os
from tqdm import tqdm
import argparse
import mmcv
import shutil


# 17 classes
colormap = np.array([[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128]])

# 17 类出现频率
class_frequency = {0:0.2882,1:0.0283,2:0.0718,3:0.6020,4:0.0097}



# 重点类别
rare_class = [4]


def preprocess(image, resize_height, resize_width):
    image = image.convert('RGB')
    image = image.resize((resize_width, resize_height))
    np_image = np.array(image)
    # HWC -> CHW
    np_image = np_image.transpose(2, 0, 1)  # CxHxW
    # normalize the image
    mean_vec = np.array([0.485, 0.456, 0.406])
    std_vec = np.array([0.229, 0.224, 0.225])
    norm_img_data = np.zeros(np_image.shape).astype('float32')
    for i in range(np_image.shape[0]):
        norm_img_data[i, :, :] = (np_image[i, :, :] / 255 - mean_vec[i]) / std_vec[i]
    np_image = np.expand_dims(norm_img_data, axis=0)  # 1xCxHxW
    return np_image


def get_predictions_from_ONNX(onnx_session, img_data):
    sess_input = onnx_session.get_inputs()
    sess_output = onnx_session.get_outputs()
    # predict with ONNX Runtime
    output_names = [output.name for output in sess_output]
    mask = onnx_session.run(output_names=output_names,
                            input_feed={sess_input[0].name: img_data})
    return mask


def get_session_inputs_outputs(session):
    sess_input = session.get_inputs()
    sess_output = session.get_outputs()
    print(f"No. of inputs : {len(sess_input)}, No. of outputs : {len(sess_output)}")

    input_names = []
    input_shapes = []
    # get input information
    for idx, input_ in enumerate(range(len(sess_input))):
        input_name = sess_input[input_].name
        input_shape = sess_input[input_].shape
        input_type = sess_input[input_].type
        print(f"{idx} Input name : {input_name}, Input shape : {input_shape}, \
        Input type  : {input_type}")
        input_names.append(input_name)
        input_shapes.append(input_shape)

    output_names = []
    output_shapes = []
    # get output information
    for idx, output in enumerate(range(len(sess_output))):
        output_name = sess_output[output].name
        output_shape = sess_output[output].shape
        output_type = sess_output[output].type
        print(f" {idx} Output name : {output_name}, Output shape : {output_shape}, \
        Output type  : {output_type}")
        output_names.append(output_name)
        output_shapes.append(output_shape)

    return input_names, input_shapes, output_names, output_shapes


def add_img_mask(img, mask, alpha=0.5):
    if mask.mode != 'RGB':
        mask = mask.convert('RGB')

    dst = Image.blend(img, mask, alpha)

    return dst

def is_video_file(path):
    # 列出一些常见的视频文件扩展名
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.mpeg', '.mpg']
    # 获取路径的后缀名并检查是否在视频扩展名列表中
    return os.path.splitext(path)[1].lower() in video_extensions


def check_if_video(dir_path):
    # 检查指定目录下是否有视频文件
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if is_video_file(file):
                return True
    return False


def calculate_metrics_filter(image, scores, ids, scores_threshold=0.95, class_frequency=None, rare_class=None):
    """
    计算图像的置信度、边缘重合度以及筛选规则
    parameters:
        image: 输入图像
        scores: 每个像素的置信度
        ids: 每个像素的类别
    return:
        is_filtered: 是否是问题图像, True or False
        unsure: 低置信度区域的类别加权不确定度
        edge: 边缘重合度
    """
    h, w = scores.shape[0], scores.shape[1]
    image = cv2.resize(image, (w, h))
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # 低置信度区域的类别加权不确定度
    area = scores < scores_threshold
    weights = np.zeros_like(ids).astype(np.float32)
    for k in class_frequency.keys():
        weights[ids == int(k)] = 1 - class_frequency[k]
    unsure = np.sum((1 - scores) * weights * area) / (h * w)

    # 边缘重合度
    ids_edge = cv2.Canny(ids.astype(np.uint8), 1, 10)
    gray_image = cv2.medianBlur(gray_image, 5)
    img_edge = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    img_edge = 255 - img_edge
    same_edge = img_edge & ids_edge
    edge = np.sum(same_edge) / (np.sum(ids_edge) + 1e-6)

    # 筛选规则
    # 情况1：该图像中出现较少出现的类别，严格筛选条件
    if len([k for k in rare_class if k in np.unique(ids)]) > 0:
        if unsure > 0.025 or (edge < 0.25 and unsure > 0.01):
            return True, unsure, edge
    # 情况2：正常筛选
    else:
        if unsure > 0.05 or (edge < 0.15 and unsure > 0.01):
            return True, unsure, edge
    return False, unsure, edge



def infer_media(args, is_video=True):
    """
    对图片文件夹/视频进行推理
    args:
        model_2_path: 模型2路径
        media_path: 图片文件夹/视频路径
        result_dir: 结果保存路径
    """
    model_2_path = args.onnx_path
    media_path = args.folder
    result_dir = args.out

    result_dir = Path(result_dir) / Path(media_path).stem
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    print('result saving dir: ', str(result_dir))

    result_vis_dir = result_dir / ('filtered_vis_' + Path(model_2_path).stem)
    result_img_dir = result_dir / ('filtered_img_' + Path(model_2_path).stem)
    for dir_path in [result_vis_dir, result_img_dir]:
        if os.path.exists(str(dir_path)):
            shutil.rmtree(str(dir_path))
        os.makedirs(str(dir_path))

    media_path = Path(media_path)
    if is_video:
        media_items = mmcv.VideoReader(str(media_path))
        print('processing video: ' + media_path.name)
    else:
        media_items = list(media_path.iterdir())
        print('processing images from: ' + str(media_path))

    pbar = tqdm(desc='processing frames' if is_video else 'processing images', total=len(media_items))
    results, resize_height, resize_width, input_names, input_shapes = None, None, None, None, None
    session, output_names = None, None
    should_print = True

    for idx, item in enumerate(media_items):
        pbar.update(1)
        if idx == 0:
            session = onnxruntime.InferenceSession(model_2_path, providers=['CUDAExecutionProvider'])
            session = onnxruntime.InferenceSession(model_2_path, providers=['CPUExecutionProvider'])
            input_names, input_shapes, output_names, output_shapes = get_session_inputs_outputs(session)
            resize_height, resize_width = input_shapes[0][2], input_shapes[0][3]

        if is_video:
            rgb_cv_image = cv2.cvtColor(item, cv2.COLOR_BGR2RGB)
            img_arr = Image.fromarray(rgb_cv_image)
        else:
            rgb_cv_image = mmcv.imread(str(item), channel_order='rgb')
            img_arr = Image.fromarray(rgb_cv_image)

        input_img = preprocess(img_arr, resize_height, resize_width)
        inputs = {input_names[0]: input_img}

        if idx == 0:
            for i in range(1, len(input_names)):
                inputs[input_names[i]] = np.zeros((input_shapes[i])).astype(np.float32)
        else:
            if args.series:
                if should_print:
                    print('--------------时序推理, 复制输入------------------')
                    should_print = False
                for i in range(1, len(input_names)):
                    inputs[input_names[i]] = results[i].reshape(input_shapes[i])
            else:
                if should_print:
                    print('---------------不进行时序推理, 除了输入1全置为0---------------')
                    should_print = False
                for i in range(1, len(input_names)):
                    inputs[input_names[i]] = np.zeros((input_shapes[i])).astype(np.float32)

        results = session.run(output_names=output_names, input_feed=inputs)

        # print(results[0])
        # 解析results得到分割概率图mask, 置信度scores, 类别ids
        mask = np.squeeze(np.array(results[0]))
        scores = np.max(mask, axis=0)
        # print(mask)
        pred_ids = np.argmax(mask, axis=0)
        # print(mask)
        # print(pred_ids)

        # 得到过滤结果
        if_filtered, unsure, edge = calculate_metrics_filter(rgb_cv_image, scores, pred_ids, 
                                                         scores_threshold=0.95, class_frequency=class_frequency, rare_class=rare_class)
        
        # print(f'{item.stem} filtered, unsure={unsure}, edge={edge} pred_ids={pred_ids} scores={scores}')
        if args.filter and not if_filtered:
            continue

        # 保存预测图
        pred_img = Image.fromarray(np.array(pred_ids).astype(np.uint8))
        pred_img.putpalette(colormap.astype(np.uint8).flatten())
        img_arr = img_arr.resize((resize_width, resize_height))
        dst = add_img_mask(img_arr, pred_img)
        if dst.mode != 'RGB':
            dst = dst.convert('RGB')
        # 保存原图
        if is_video:
            save_path = str(result_img_dir) + '/' + media_path.stem + '_{:06d}.jpg'.format(idx)
            print(f'saving img to {save_path}')
            img_arr.save(save_path)
            save_path = str(result_vis_dir) + '/' + media_path.stem + '_{:06d}.jpg'.format(idx)
            print(f'saving result to {save_path}')
            dst.save(save_path)
        else:
            save_path = str(result_img_dir) + '/' + str(item.name)
            print(f'saving img to {save_path}')
            shutil.copyfile(str(item), save_path)
            print(f'saving result to {save_path}')
            save_path = str(result_vis_dir) + '/' + str(item.name)
            dst.save(save_path)

    pbar.close()


def parse_args():
    parser = argparse.ArgumentParser(description='Process some integers')
    parser.add_argument('onnx_path', type=str,
                        help='video 2 onnx')
    parser.add_argument('folder', type=str,
                        help='image folder path/video folder path')
    parser.add_argument('-o', '--out', default='./results', type=str,
                        help='pred dir')
    parser.add_argument('-s', '--series', default=False, type=bool, help="是否进行ONNX时序推理")
    parser.add_argument('-f', '--filter', default=False, type=bool, help="是否进行问题图片过滤")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    input_path = args.folder
    if os.path.isdir(input_path):
        print("推理文件夹。。。")
        if check_if_video(input_path):
            print("推理视频。。。")
            for path in Path(input_path).iterdir():
                args.folder = str(path)
                infer_media(args, is_video=True)
        else:
            print("推理图片。。。")
            infer_media(args, is_video=False)
    else:
        print("推理单个视频。。。")
        infer_media(args, is_video=True)


if __name__ == '__main__':
    main()
