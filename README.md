# 项目介绍

这个项目是多分类的任务，即一张图片该图片只属于一类。
这个任务是7大类和41小类的数据分类，是用4种不同的模型对口腔图片进行分类，
这四种模型分别为VGG16，ResNet152，DenseNet201，Googlenet，其中7大类和41小类又进行了数据增强对比实验。
一共有16组实验。

# 数据介绍
7大类无数据增强和41小类无数据增强的pic
J:\yinda\zhujiaxi\data\teeth_png_zong(准备上传百度网盘)
一共有3818张
数据增强之后一共11150



其中训练集测试集验证集的划分是根据csv表格来的，即根据csv表格中的内容来确定哪些是训练集，测试集和验证集。

# 已有环境
Windows操作系统在local10的conda环境名为zhujiaqian。
Linux操作系统在local3上，环境名为zhujiaqia

# 项目运行
1.修改总分类图片路径，在mypath.py文件中，output_dir

2.修改训练集，测试集和验证集路径,在dataset.py中，_load_dataset函数中，其中zcx-val是验证集路径


3.修改weights_group_1
在train.py文件中
这个是每个类别的权重系数用于loss计算（7类就写7个1，41类就写41个1）

4.num_classes，
在train.py文件中，根据任务不同写不同的数字

5.修改保存模型的文件名字
在train.py文件中

6.修改对应的模型
用torchvision包的内置的四个经典模型，之后修改模型输出的类别数


# 根据任务选择不同的数据

原始数据3818，适用于非数据增强实验的8组实验，即VGG16，ResNet152，DenseNet201，Googlenet这四种模型情况以及7类和41类的情况。

数据增强11150，适用于数据增强实验的8组实验，即VGG16，ResNet152，DenseNet201，Googlenet这四种模型情况以及7类和41类的情况。

存在4种excle文件，每种excle文件可以用于4种实验(这四种就是4种需要跑的模型)，这四种excle分别是7类无数据增强，41类无数据增强，7类数据增强，41类无数据增强。

所以在设置实验的时候需要仔细查看数据来源，excle和model是否设置的是你当前做的实验


# 数据路径
在local3节点中，准备上传百度网盘
数据增强图片路径/home/user/yinda/zhujiaqian/densenetdataup/project/data/data

无数据增强图片数据/home/user/yinda/zhujiaqian/data/data/teeth_png_zong

无数据增强7类excle路径/home/user/yinda/zhujiaqian/excle3

无数据增强41类excle路径/home/user/yinda/zhujiaqian/class41/dataexcle


数据增强7类excle路径
/home/user/yinda/zhujiaqian/densenetdataup/project/data/dataexcle/trian.xlsx
Val路径/home/user/yinda/zhujiaqian/excle3/val.xlsx
text路径/home/user/yinda/zhujiaqian/excle3/text.xlsx