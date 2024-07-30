import os
from pathlib import Path
from argparse import ArgumentParser

#python rename.py ./solve ./solve_rename --prefix 0705_lixun_

def files_rename(img_dir, prefix, dst_dir):
    # 遍历并排序图片文件
    img_list = sorted(img_dir.iterdir())
    
    print(f'The number of images: {len(img_list)}')
    
    for index, img_path in enumerate(img_list):
        # if img_path.suffix.lower() not in ['.jpg', '.png']:
        #     print(f'Not a JPG or PNG image: {img_path.name}')
        #     continue
        
        # 构建新文件名
        new_name = f'{prefix}{str(index).zfill(6)}{img_path.suffix}' if prefix else f'{str(index).zfill(6)}{img_path.suffix}'
        new_path = dst_dir / new_name
        
        # 重命名文件
        img_path.rename(new_path)
    
    print(f'The number of renamed images: {index + 1}')

def parse_args():
    parser = ArgumentParser(description='Rename files in a directory.')
    parser.add_argument('folder_path', type=str,  help='Path to the source folder containing images.')
    parser.add_argument('dst_folder', type=str, help='Destination folder path for renamed images.')
    parser.add_argument('--prefix', type=str, default='', help='Optional prefix to add to the new filenames.')
    return parser.parse_args()

def main():
    args = parse_args()
    src_dir = Path(args.folder_path).resolve()
    dst_dir = Path(args.dst_folder)
    
    # 确保目标目录存在
    dst_dir.mkdir(parents=True, exist_ok=True)
    assert src_dir.is_dir(), 'Source is not a valid directory.'
    
    files_rename(src_dir, args.prefix, dst_dir)

if __name__ == '__main__':
    main()
