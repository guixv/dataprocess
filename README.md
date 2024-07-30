# CVAT

python CVAT_RGB_mask_to_train_mask.py E:\python\mmsegCode\data/0719 E:/python/dataconvert/label_merge_20240705.txt

python CVAT_RGB_mask_to_train_mask.py E:\python\mmsegCode\data/0719/VOCCV_0719 E:/python/dataconvert/label_merge_20240725_6.txt

python pytorch2onnx.py --config /data/work_folder/qiuchaoyi/code/mmsegCode/configs/botseg/baseline.py --checkpoint /data/work_folder/qiuchaoyi/code/mmsegCode/work_dirs/baseline/iter_80000.pth  --shape 256 512
python3 -m onnxsim tmp.onnx out.onnx

python video_onnx_infer_2.py hard_0715.onnx temp --concat --save-img --pr preds/todo

python rename.py ./solve ./solve_rename --prefix 0705_li_
