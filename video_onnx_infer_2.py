import onnxruntime
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import os
from tqdm import tqdm
import argparse
import scipy.io as scio
import onnx
from PIL import ImageOps, ImageDraw, ImageFont
import mmcv
import shutil

#python video_onnx_infer_2.py out.onnx newcv_0704_hik3_floor1.mp4 --concat --save-img --pr preds/hard

def preprocess(image, resize_height, resize_width):
    """perform pre-processing on raw input image

    :param image: raw input image
    :type image: PIL image
    :param resize_height: resize height of an input image
    :type resize_height: Int
    :param resize_width: resize width of an input image
    :type resize_width: Int
    :return: pre-processed image in numpy format
    :rtype: ndarray of shape 1xCxHxW
    """
    image = image.convert('RGB')
    image = image.resize((resize_width, resize_height))
    # image = ImageOps.equalize(image)
    np_image = np.array(image)

    #########################
    # cv_img = cv2.cvtColor(np.array(np_image), cv2.COLOR_RGB2BGR)
    # b, g, r = cv2.split(cv_img)
    # # 对 R 通道应用 CLAHE
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # b = clahe.apply(b)
    # g = clahe.apply(g)
    # r = clahe.apply(r)

    # # 合并通道
    # enhanced_image = cv2.merge([b, g, r])
    
    # cv_img = cv2.cvtColor(np.array(enhanced_image), cv2.COLOR_BGR2RGB)
    # np_image = np.array(cv_img)

    ###################################

    # # 转换为 HSV 空间
    # hsv = cv2.cvtColor(np.array(np_image), cv2.COLOR_RGB2HSV)
    # # 分离通道
    # h, s, v = cv2.split(hsv)
    # # 对亮度通道应用 CLAHE
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # v = clahe.apply(v)
    # # 合并通道
    # hsv = cv2.merge([h, s, v])
    # # 转换回 RGB 空间
    # cv_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    # np_image = np.array(cv_img)
    
    #####################################

    # # 转换为 YUV
    # yuv = cv2.cvtColor(np.array(np_image), cv2.COLOR_RGB2YUV)
    # # 分离 Y 通道
    # y, u, v = cv2.split(yuv)
    # # 对 Y 通道应用 CLAHE
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # y = clahe.apply(y)
    # # 合并通道
    # yuv = cv2.merge([y, u, v])
    # # 转换回 RGB
    # cv_img = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)
    # np_image = np.array(cv_img)

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
    """perform predictions with ONNX Runtime

    :param onnx_session: onnx model session
    :type onnx_session: class InferenceSession
    :param img_data: pre-processed numpy image
    :type img_data: ndarray with shape 1xCxHxW
    :return: boxes, labels , scores , masks with shapes
            (No. of instances, 4) (No. of instances,) (No. of instances,)
            (No. of instances, 1, HEIGHT, WIDTH))
    :rtype: tuple
    """

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


def concat_img_mask(img, mask):
    # mask = Image.fromarray(mask)
    if mask.mode != 'RGB':
        mask = mask.convert('RGB')

    dst = Image.new('RGB', (img.width + img.width, img.height))
    dst.paste(img, (0, 0))
    dst.paste(mask, (img.width, 0))

    return dst

def concat_img_mask_with_index(img, mask, index):
    # 确保mask是RGB模式，以便与原图拼接
    if mask.mode != 'RGB':
        mask = mask.convert('RGB')

    # 计算序号文本的宽度，以便确定画布的总宽度
    draw = ImageDraw.Draw(img)
    # font = ImageFont.truetype("arial.ttf", size=30)  # 使用系统中可用的字体文件，和字号大小
    text_width, _ = draw.textsize(str(index))
    total_width = img.width + max(mask.width, img.width + text_width)
    dst = Image.new('RGB', (total_width, img.height))

    # 粘贴原图
    dst.paste(img, (0, 0))

    # 在原图右侧添加序号
    draw = ImageDraw.Draw(dst)
    draw.text((img.width, 0), str(index), fill=(255, 255, 255))  # 白色序号

    # 粘贴掩码图
    dst.paste(mask, (img.width + text_width, 0))  # 考虑到序号的宽度，适当调整掩码的起始位置

    return dst
    
def add_img_mask(img, mask, alpha=0.5):
    if mask.mode != 'RGB':
        mask = mask.convert('RGB')
    
    dst = Image.blend(img, mask, alpha)
    
    return dst


# 23 classes
colormap = np.array([[73,223,202],[245,32,90], [164,253,9], [13,243,181], [204,153,51],
            [36,179,83], [0,0,0], [7,7,43], [95,71,100], [187,167,33],
            [205,14,34], [255,0,124], [42,79,247], [0,92,56], [131,224,112], 
            [80,192,0], [140,120,240], [161,79,224], [255,106,77], [248,108,254],
            [233,206,108], [21,21,224], [255, 255, 253]])
            
# 17
colormap = np.array([[245,32,90], [164,253,9], [204,153,51], [36,179,83], [102,255,102],
                [7,7,43], [95,71,100], [255,0,124], [42,79,247], [131,224,112],
                [140,120,240], [161,79,224], [255,106,77], [248,108,254], [233,206,108], 
                [21,21,224], [66,74,47]])


def infer_imgs(args):
    model_2_path = args.m2
    images_dir = args.im
    result_dir = args.pr
    concat_mask = args.concat
    add_mask = args.add
    save_int8 = args.save_int8

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    result_dir = Path(result_dir)

    print('result saving dir: ', str(result_dir))
    images_dir = Path(images_dir)
    images_list = list(images_dir.iterdir())

    input_shape = None
    pbar = tqdm(len(list(images_dir.iterdir())))
    for idx, img_path in enumerate(images_list):
        if idx == 0:
            session = onnxruntime.InferenceSession(model_2_path, )#providers=['CUDAExecutionProvider'])
            input_names, input_shapes, output_names, output_shapes = get_session_inputs_outputs(session)
            resize_height, resize_width = input_shapes[0][2], input_shapes[0][3]

        img = Image.open(str(img_path))
        #########################################################
        # img = img.crop((32, 30, 608, 420))

        ######################################+++++++++++++++++++++++++++++++++++++++
        image_width, image_height = img.size
        img_data = preprocess(img, resize_height, resize_width)

        inputs = {input_names[0]: img_data}
        
        if idx == 0:  #第一帧初始化为0
            for i in range(1, len(input_names)):
                inputs[input_names[i]] = np.zeros((input_shapes[i])).astype(np.float32)
        else:
            if False:
                ################## 除了输入1全置为0 ##################
                for i in range(1, len(input_names)):
                    inputs[input_names[i]] = np.zeros((input_shapes[i])).astype(np.float32)

            else:
                ################## 复制输入 ##################
                # for i in range(1, len(input_names)):
                    # inputs[input_names[i]] = results[i].reshape(input_shapes[i]) 
                
                pass

        results = session.run(output_names=output_names,
                            input_feed=inputs)

        mask = results[0]

        mask = np.array(mask).astype(np.uint8)
        mask = np.squeeze(mask)
        mask = Image.fromarray(mask)
        # put color palette
        # mask.putpalette(colormap.astype(np.uint8).flatten())

        img = img.resize((resize_width, resize_height))

        if args.save_img:
            input_file = str(Path(result_dir)) + '_{:06d}.jpg'.format(idx)
            img.save(input_file)

        # # 转换为 YUV
        # yuv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2YUV)
        # # 分离 Y 通道
        # y, u, v = cv2.split(yuv)
        # # 对 Y 通道应用 CLAHE
        # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        # y = clahe.apply(y)
        # # 合并通道
        # yuv = cv2.merge([y, u, v])
        # # 转换回 RGB
        # cv_img = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)
        # img = np.array(cv_img)
        # img = Image.fromarray(img)

        if add_mask:
            mask.putpalette(colormap.astype(np.uint8).flatten())
            dst = add_img_mask(img, mask)
        elif concat_mask:
            mask.putpalette(colormap.astype(np.uint8).flatten())
            dst = concat_img_mask(img, mask)
        elif save_int8:
            dst = mask
        else:
            mask.putpalette(colormap.astype(np.uint8).flatten())
            dst = mask 
        
        dst.save(str(result_dir / img_path.stem) + '.png')
        pbar.update(1)


def infer_video(args):
    model_2_path = args.m2
    video_path = args.im
    result_dir = args.pr
    concat_mask = args.concat
    add_mask = args.add
    save_int8 = args.save_int8
    output_name = args.output

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    result_dir = Path(result_dir)
    print('result saving dir: ', str(result_dir))

    video_path = Path(video_path)

    # 创建测试保存文件夹，文件夹名为视频名字
    result_imgs_dir = result_dir / (video_path.stem + '_' + Path(model_2_path).stem)
    if not os.path.isdir(str(result_imgs_dir)):
        os.makedirs(str(result_imgs_dir))
    
    if args.save_img:
        input_imgs_dir = result_dir / (video_path.stem + '_input')
        if not os.path.isdir(str(input_imgs_dir)):
            os.makedirs(str(input_imgs_dir))

    # 读取视频
    video = mmcv.VideoReader(str(video_path))
    print('processing video: ' + video_path.name)

    idx=0
    should_print=True
    pbar = tqdm(desc='processing frames', total = len(video))
    for img in video:
        pbar.update(1)
        if idx == 0:
            session = onnxruntime.InferenceSession(model_2_path, )#providers=['CUDAExecutionProvider'])
            input_names, input_shapes, output_names, output_shapes = get_session_inputs_outputs(session)
            resize_height, resize_width = input_shapes[0][2], input_shapes[0][3]
        
        rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        original_img = Image.fromarray(rgb_image)
        
        ########################################################################################   裁剪 ################
        original_img = original_img.crop((30, 15, 610, 455))
        # original_img = original_img.crop((32, 160, 608, 448))
        #############################################++++++++++++++++++++++++++++++++++++++++++
        
        img_data = preprocess(original_img, resize_height, resize_width)

        inputs = {input_names[0]: img_data}
        
        if idx == 0:  #第一帧初始化为0
            for i in range(1, len(input_names)):
                inputs[input_names[i]] = np.zeros((input_shapes[i])).astype(np.float32)
        else:
            if False:
                ################## 除了输入1全置为0 ##################
                if should_print:
                    print('----------------除了输入1全置为0---------------')
                    should_print = False 
                for i in range(1, len(input_names)):
                    inputs[input_names[i]] = np.zeros((input_shapes[i])).astype(np.float32)

            else:
                ################## 复制输入 ##################
                if should_print:
                    print('--------------复制输入------------------')
                    should_print = False 
                for i in range(1, len(input_names)):
                    inputs[input_names[i]] = results[i].reshape(input_shapes[i]) 
                pass

        results = session.run(output_names=output_names,
                            input_feed=inputs)

        mask = results[0]
        mask = np.array(mask).astype(np.uint8)
        mask = np.squeeze(mask)
        mask = Image.fromarray(mask)

        resized_img = original_img.resize((resize_width, resize_height))
        mask = mask.resize((resize_width, resize_height))
        if add_mask:
            mask.putpalette(colormap.astype(np.uint8).flatten())
            dst = add_img_mask(resized_img, mask)
        elif concat_mask:
            mask.putpalette(colormap.astype(np.uint8).flatten())
            dst = concat_img_mask_with_index(resized_img, mask, '{:06d}'.format(idx))
        elif save_int8:
            dst = mask
        else:
            mask.putpalette(colormap.astype(np.uint8).flatten())
            dst = mask 

        if dst.mode != 'RGB':
            dst = dst.convert('RGB')
        output_file = str(result_imgs_dir) + '/' + '{:06d}.jpg'.format(idx)
        dst.save(output_file)
        if args.save_img:
            input_file = str(input_imgs_dir) + '/' + video_path.stem + '_{:06d}.jpg'.format(idx)
            original_img.save(input_file)

        idx = idx + 1   
    if output_name == '':
        outpath = str(result_dir) + '/' + str(video_path.stem) + '_' + str(Path(model_2_path).stem) + '.mp4'
        print("saving video to {}".format(outpath))
        mmcv.video.frames2video(str(result_imgs_dir), outpath, fps=15, fourcc='mp4v')
    else:
        outpath = str(result_dir) + '/' + output_name + '.mp4'
        print("saving video to {}".format(outpath))
        mmcv.video.frames2video(str(result_imgs_dir), outpath, fps=15, fourcc='mp4v')

    # 删除图片
    shutil.rmtree(str(result_imgs_dir))


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


def parse_args():
    parser = argparse.ArgumentParser(description='Process some integers')
    parser.add_argument('m2',  default='./video_2.onnx', type=str,
                        help='video 2 onnx')
    parser.add_argument('im', default='./video_img', type=str,
                   help='image dir')
    parser.add_argument('--pr', default='./preds', type=str,
                   help='pred dir') 
    parser.add_argument('--output', default='', type=str,
                   help='output file name for video infer')                    
    parser.add_argument('--concat', action='store_true', 
                   help='concat img and mask') 
    parser.add_argument('--add', action='store_true', help='add img and mask') 
    parser.add_argument('--save-img', action='store_true', help='save img')
    parser.add_argument('--save-int8', action='store_true', help='save int8 pred mask') 
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    input_path = args.im
    
    if os.path.isdir(input_path):
        print("images_dir 是一个文件夹。")
        if check_if_video(input_path):
            print("文件夹中包含视频文件。")
            for path in Path(input_path).iterdir():
                args.im = str(path)
                infer_video(args)
        else:
            print("文件夹中为图片")
            infer_imgs(args)
    else:
        print("处理单个视频")
        infer_video(args)


    # 保存为.npz格式文件
    # np.savez(str(result_dir / img_path.stem) + '.npz', output=np.array(results[-1]))
            

if __name__ == '__main__':
    main()


    
