import os
import shutil
import argparse

#python copy_images_by_last_digits.py ./preds/newcv_0704_hik3_floor1_input ./solve --digits 001831 001859                  

def pad_number(num, width=6):
    """将数字补足到指定宽度，不足部分前面补零"""
    return str(num).zfill(width)

def copy_images_by_last_digits(source_folder, target_folder, digits_list, digit_count=6):
    """
    根据图片文件名末尾的数字从源文件夹拷贝图片到目标文件夹。
    
    :param source_folder: 图片源文件夹路径
    :param target_folder: 目标文件夹路径
    :param digits_list: 需要拷贝的图片文件名末尾数字列表
    :param digit_count: 文件名中用于匹配的数字位数，默认为4位
    """
    new_digits_list = [pad_number(num, width=digit_count) for num in digits_list]
    # 确保目标文件夹存在
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
        
    # 遍历源文件夹
    for filename in os.listdir(source_folder):
        # 检查文件是否为图片（这里简化处理，只检查是否以常见图片格式结尾）
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
            # 提取文件名末尾的数字
            try:
                base_name = os.path.basename(filename)
                file_name_without_ext = os.path.splitext(base_name)[0]  # 分离并获取不带扩展名的部分
                last_digits = file_name_without_ext[-digit_count:]  # 假设数字在文件名末尾
                # 如果数字在列表中，则拷贝文件
                if last_digits in new_digits_list:
                    src_path = os.path.join(source_folder, filename)
                    dst_path = os.path.join(target_folder, filename)
                    shutil.copy2(src_path, dst_path)
                    print(f"Copied: {filename}")
            except ValueError:
                # 忽略非数字结尾的文件
                pass

def main():
    # 创建解析器
    parser = argparse.ArgumentParser(description="Copy images based on the last digits of their names.")
    
    # 添加命令行参数
    parser.add_argument("source_folder", help="Path to the source folder containing images.")
    parser.add_argument("target_folder", help="Path to the target folder where images will be copied.")
    parser.add_argument("--digits", nargs='+', type=int, default=[], help="List of last digits to match for copying images.")
    parser.add_argument("--digit_count", type=int, default=6, help="Number of digits to consider at the end of filenames. Defaults to 6.")
    
    # 解析参数
    args = parser.parse_args()

    # 调用函数
    copy_images_by_last_digits(args.source_folder, args.target_folder, args.digits, args.digit_count)

if __name__ == "__main__":
    main()


