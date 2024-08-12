import os
import random
#######随机切分训练验证集
if __name__ == '__main__':
    #设置随机数种子，复习随机场景所必需的
    random.seed(10)
    #获取image路径
    imagepath = "E:\python\mmsegCode\data/VOCCV_0807/"

    total_image = os.listdir(imagepath + "JPEGImages/")
    #print(len(total_image))
    tr = int(len(total_image)*0.95) #用于训练数据集,百分之70
    va = int(len(total_image)*0.05) #验证
    te = int(len(total_image)*0.0) #测试

    indices = list(range(len(total_image))) #获得迭代类型0-num
    train=random.sample(indices,tr)  #百分之70*总数量等于多少，用来训练的样本的索引
    val = random.sample(indices,va)
    test = random.sample(indices,te)

    #打开三个文件，需要手动在自己的目录下创建三个txt文件
    ftrain = open(imagepath + "ImageSets/Segmentation/train.txt",'w')
    fval = open(imagepath + "ImageSets/Segmentation/val.txt",'w')
    ftest = open(imagepath + "ImageSets/Segmentation/test.txt",'w')

    for i in indices:
        name = total_image[i].split(".")[0]  #截取图像名，去掉.jpg
        if i in train:
            ftrain.write( name  +"\n")
        if i in val:
            fval.write(name +"\n")
        if i in test:
            ftest.write(name +"\n")
    ftrain.close()
    fval.close()
    ftest.close()
