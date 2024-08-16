# CVAT

python CVAT_RGB_mask_to_train_mask.py E:\python\mmsegCode\data/0719/VOCCV_0719 E:/python/dataconvert/label_merge_20240725_6.txt

python pytorch2onnx.py --config /data/work_folder/qiuchaoyi/code/mmsegCode/configs/botseg/maskconver_starnet_Full_20k.py --checkpoint /data/work_folder/qiuchaoyi/code/mmsegCode/work_dirs/maskconver_starnet_Full_20k/0815_1.pth  --shape 256 512
maskconver_starnet_Full_20k

python pytorch2onnx.py --config /data/work_folder/qiuchaoyi/code/mmsegCode/configs/botseg/baseline.py --checkpoint /data/work_folder/qiuchaoyi/code/mmsegCode/work_dirs/baseline/iter_80000.pth  --shape 256 512
python3 -m onnxsim tmp.onnx out.onnx

python video_onnx_infer_2.py hard_0715.onnx temp --concat --save-img --pr preds/todo

python rename.py ./solve ./solve_rename --prefix 0812_hik5_floor11_mat_000145_

pkill -u <username> -f python

pip install scikit-image numba

python onnx_infer_filter.py measure.onnx E:\python\outseg\cv -o E:\python\outseg\preds\todo\cv_select -f True
##Note 需要在封装onnx的过程中把最后输出前的argmax给删除，否则输出会是channel=1的模型
