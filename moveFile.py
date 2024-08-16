
import os
import shutil
from pathlib import Path

def remove_extension(filename):
    """移除文件名中的扩展名"""
    return os.path.splitext(filename)[0]

def copy_matching_files(src_dir, ref_dir, dest_dir):
    """
    将src_dir中文件名（去除后缀）与ref_dir中文件名相同的文件复制到dest_dir
    :param src_dir: 源文件夹路径
    :param ref_dir: 参考文件夹路径，用于比较文件名（去除后缀）
    :param dest_dir: 目标文件夹路径
    """
    # 确保目标文件夹存在
    Path(dest_dir).mkdir(parents=True, exist_ok=True)

    # 遍历源文件夹中的所有文件
    for src_file in os.listdir(src_dir):
        src_file_path = os.path.join(src_dir, src_file)
        
        # 如果src_file是文件（避免目录）
        if os.path.isfile(src_file_path):
            # 去除扩展名
            src_file_base = remove_extension(src_file)
            
            # 检查在参考文件夹中是否存在相同名称（去除后缀）的文件
            for ref_file in os.listdir(ref_dir):
                if remove_extension(ref_file) == src_file_base:
                    # 构造目标文件路径
                    dest_file_path = os.path.join(dest_dir, src_file)
                    
                    # 复制文件
                    shutil.copy(src_file_path, dest_file_path)
                    print(f"Copied: {src_file_path} -> {dest_file_path}")
                    break  # 假设每个基名只对应一个参考文件，复制后跳出循环

if __name__ == '__main__':
    # 使用示例
    src_folder = 'A'  # A文件夹路径
    ref_folder = 'B'  # B文件夹路径
    dest_folder = 'C' # C文件夹路径

    copy_matching_files(src_folder, ref_folder, dest_folder)
