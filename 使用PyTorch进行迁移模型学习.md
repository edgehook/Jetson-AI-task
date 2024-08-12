
## Transfer Learning with PyTorch

迁移学习是一种在新的或者自定义数据集上重新训练模型的技术，它比从头开始训练网络所需的时间更少，可以对预训练模型的权重进行微调，以对自定义数据集进行分类。本例将基于Jetson AGX Orin平台使用SSD-Mobilenet骨干网络进行模型训练。

Pytorch是我们将要使用的机器学习框架，下面提供了搭建pytorch环境、训练代码以及示例数据集，使我们能够重新训练 Jetson 上的各种网络，以开始训练和部署我们自己的模型。

### 安装Pytorch
下载[Pytorch for Jetson](https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048)，使用pip3工具下载软件包【加-i参数可以指定国内下载源】
``` bash
apt install python3-pip libopenblas-dev
 
python3 -m pip install aiohttp opencv-python scipy=='1.5.3' -i https://pypi.tuna.tsinghua.edu.cn/simple
 
pip3 install ./torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl -i https://pypi.tuna.tsinghua.edu.cn/simple
 
# 编译torchvision所需依赖包
apt install ffmpeg libavutil-dev libavcodec-dev libavformat-dev 
libavdevice-dev libavfilter-dev libswscale-dev libswresample-dev 
libswresample-dev libpostproc-dev libjpeg-dev libpng-dev
 
# 下载torchvision源码
git clone https://github.com/pytorch/vision torchvision
 
# 编译torchvision
python3 setup.py install 
 
# 利用编译成功的源码生成*.whl文件，生成文件路径在/源码根目录/dist/
python3 setup.py bdist_wheel    
```
---
### 验证 PyTorch
``` bash
>>> import torch
>>> print(torch.__version__)
>>> print('CUDA available: ' + str(torch.cuda.is_available()))
>>> a = torch.cuda.FloatTensor(2).zero_()
>>> print('Tensor a = ' + str(a))
>>> b = torch.randn(2).cuda()
>>> print('Tensor b = ' + str(b))
>>> c = a + b
>>> print('Tensor c = ' + str(c))
 
>>> import torchvision
>>> print(torchvision.__version__)
```

### 下载训练代码
本例使用[pytorch-ssd](./pytorch-ssd/)作为训练基础代码，此 repo 使用 MobileNet 主干在 PyTorch 中实现了SSD（单次多框检测器），用于物体检测。它还具有对 Google Open Images 数据集和 Pascal VOC 进行再训练的支持。
``` bash
pytorch-ssd 目录结构
├── data/             #放置数据集目录
├── eval_ssd.py
├── LICENSE
├── models/           #放置预训练模型，以及重新生成的新模型
├── onnx_export.py    #模型转换代码，将pytorch(*.pth)转换成onnx(*.onnx) 
├── open_images_classes.txt      #open images的对象分类文件，有600种对象   
├── open_images_downloader.py    #下载open images数据集
├── README.md
├── requirements.txt
├── run_ssd_example.py
├── train_ssd.py       #训练模型的核心代码
└── vision/            #处理数据集的代码文件
    ├── datasets/
    ├── __init__.py
    ├── nn
    ├── prunning
    ├── ssd
    ├── test
    ├── transforms
    └── utils
```
以上pytorch-ssd目录中我们当前只需要关心三个，data/，models，和train_ssd.py。

1. data/：是放置数据集的路径，本例中使用的测试数据集是[CCPD](https://github.com/detectRecog/CCPD)(Chinese City Parking Dataset)，默认下载的数据结构是test/，train/，val/，需将其转换成Pascal VOC格式的数据集，网上也有已经转换好的[voc数据集](https://blog.csdn.net/qq_21386397/article/details/138757115)ps：关于[Pascal  VOC数据集格式请参考博客](https://blog.csdn.net/weixin_44095109/article/details/140990451?spm=1001.2014.3001.5502)

原始CCPD2020数据集格式：
```bash
CCPD2020
├── test/
├── train/
└── val/
```

转换后的CCPD2020 VOC数据集格式：
```bash
CCPD2020-voc
├── Annotations/  # 图片标注的xml文件
├── labels.txt   # 标签文件，必须要有 
├── JPEGImages/   # 图片放置目录
├── ImageSets/    # train.txt, test.txt(trainval.txt), val.txt放置目录
└── Main/
    ├── test.txt
    ├── train.txt
    ├── trainval.txt
    └── val.txt
```

2. models：预训练模型文件以及通过训练生成的模型文件，本例中使用预训练模型文件[mobilenet-v1-ssd-mp-0_675.pth](https://drive.google.com/drive/folders/1pKn-RifvJGWiOx0ZCRLtCXM5GT5lAluu)

3. 训练 SSD-Mobilenet 模型
运行train_ssd.py脚本启动训练：
``` bash
python3 train_ssd.py --dataset-type=voc --data=data/CCPD2020-voc --model-dir=models/CCPD2020-voc --batch-size=4 --epochs=30
 
以下是运行训练脚本时可以使用的一些常用选项：
--data		    数据集的位置(default: data/)
--dataset-type  指定数据集类型。当前支持 voc 和 open_images(default: open_images)
--model-dir		输出训练好的模型检查点的目录(default: models/)
--net           网络架构，可以是 mb1-ssd、mb1-ssd-lite、mb2-ssd-lite 或 vgg16-ssd(default:mobilenet-v1-ssd-mp-0_675.pth)
--resume		恢复训练的现有检查点的路径
--batch-size	尝试增加可用内存(default: 4)
--epochs	    最好达到 100，但会增加训练时间(default: 30)
--workers	    数据加载器线程数（0 = 禁用多线程）(default: 2)
```

### 开始训练，可以看到随着时间的推移，你会看到损失减少：
``` bash 
python3 train_ssd.py --dataset-type=voc --data=data/CCPD2020-voc --model-dir=models/voc-ccpd2020 --batch-size=4 --epochs=30
 
2024-08-09 11:16:32 - Using CUDA...
2024-08-09 11:16:32 - Namespace(balance_data=False, base_net=None, base_net_lr=0.001, batch_size=4, checkpoint_folder='models/voc-ccpd2020', dataset_type='voc', datasets=['data/CCPD2020-voc'], debug_steps=10, extra_layers_lr=None, freeze_base_net=False, freeze_net=False, gamma=0.1, log_level='info', lr=0.01, mb2_width_mult=1.0, milestones='80,100', momentum=0.9, net='mb1-ssd', num_epochs=30, num_workers=2, pretrained_ssd='models/mobilenet-v1-ssd-mp-0_675.pth', resolution=300, resume=None, scheduler='cosine', t_max=100, use_cuda=True, validation_epochs=1, validation_mean_ap=False, weight_decay=0.0005)
2024-08-09 11:16:43 - model resolution 300x300
2024-08-09 11:16:43 - SSDSpec(feature_map_size=19, shrinkage=16, box_sizes=SSDBoxSizes(min=60, max=105), aspect_ratios=[2, 3])
2024-08-09 11:16:43 - SSDSpec(feature_map_size=10, shrinkage=32, box_sizes=SSDBoxSizes(min=105, max=150), aspect_ratios=[2, 3])
2024-08-09 11:16:43 - SSDSpec(feature_map_size=5, shrinkage=64, box_sizes=SSDBoxSizes(min=150, max=195), aspect_ratios=[2, 3])
2024-08-09 11:16:43 - SSDSpec(feature_map_size=3, shrinkage=100, box_sizes=SSDBoxSizes(min=195, max=240), aspect_ratios=[2, 3])
2024-08-09 11:16:43 - SSDSpec(feature_map_size=2, shrinkage=150, box_sizes=SSDBoxSizes(min=240, max=285), aspect_ratios=[2, 3])
2024-08-09 11:16:43 - SSDSpec(feature_map_size=1, shrinkage=300, box_sizes=SSDBoxSizes(min=285, max=330), aspect_ratios=[2, 3])
2024-08-09 11:16:43 - Prepare training datasets.
2024-08-09 11:16:44 - VOC Labels read from file:  ('BACKGROUND', 'plate')
2024-08-09 11:16:44 - Stored labels into file models/voc-ccpd2020/labels.txt.
2024-08-09 11:16:44 - Train dataset size: 5006
2024-08-09 11:16:44 - Prepare Validation datasets.
2024-08-09 11:16:44 - VOC Labels read from file:  ('BACKGROUND', 'plate')
2024-08-09 11:16:44 - Validation dataset size: 5006
2024-08-09 11:16:44 - Build network.
2024-08-09 11:16:44 - Init from pretrained SSD models/mobilenet-v1-ssd-mp-0_675.pth
2024-08-09 11:16:44 - Took 0.09 seconds to load the model.
2024-08-09 11:16:44 - Learning rate: 0.01, Base net learning rate: 0.001, Extra Layers learning rate: 0.01.
2024-08-09 11:16:44 - Uses CosineAnnealingLR scheduler.
2024-08-09 11:16:44 - Start training from epoch 0.
/usr/local/lib/python3.8/dist-packages/torch/nn/_reduction.py:42: UserWarning: size_average and reduce args will be deprecated, please use reduction='sum' instead.
  warnings.warn(warning.format(ret))
2024-08-09 11:16:58 - Epoch: 0, Step: 10/1252, Avg Loss: 12.3072, Avg Regression Loss 5.4673, Avg Classification Loss: 6.8400
2024-08-09 11:16:59 - Epoch: 0, Step: 20/1252, Avg Loss: 7.8495, Avg Regression Loss 3.7893, Avg Classification Loss: 4.0602
2024-08-09 11:17:01 - Epoch: 0, Step: 30/1252, Avg Loss: 7.8789, Avg Regression Loss 3.9562, Avg Classification Loss: 3.9227
2024-08-09 11:17:03 - Epoch: 0, Step: 40/1252, Avg Loss: 6.3876, Avg Regression Loss 3.6811, Avg Classification Loss: 2.7065
2024-08-09 11:17:04 - Epoch: 0, Step: 50/1252, Avg Loss: 6.3902, Avg Regression Loss 3.0269, Avg Classification Loss: 3.3633
2024-08-09 11:17:05 - Epoch: 0, Step: 60/1252, Avg Loss: 4.7573, Avg Regression Loss 2.2380, Avg Classification Loss: 2.5193
2024-08-09 11:17:06 - Epoch: 0, Step: 70/1252, Avg Loss: 4.8878, Avg Regression Loss 2.1748, Avg Classification Loss: 2.7129
2024-08-09 11:17:08 - Epoch: 0, Step: 80/1252, Avg Loss: 4.9830, Avg Regression Loss 2.2068, Avg Classification Loss: 2.7761
```

### 将模型转换为 ONNX
需要将训练好的模型从 PyTorch 转换为 ONNX，以便我们可以用 Jetson TensorRT 加载它：
``` bash
python3 onnx_export.py --model-dir=models/voc-ccpd2020
```
这将在pytorch-ssd/models/voc-ccpd2020/保存ssd-mobilenet.onnx的模型.

### 使用 TensorRT 处理图像
为了对一些静态测试图像进行分类，我们将使用扩展的命令行参数detectnet（或detectnet.py）来加载我们的自定义 SSD-Mobilenet ONNX 模型。
```bash
./detectnet --model=models/voc-ccpd2020/ssd-mobilenet.onnx \
--labels=models/voc-ccpd2020/labels.txt --input-blob=input_0 \
--output-cvg=scores --output-bbox=boxes "./test/*.jpg" ./demo/plate_%i.jpg
```
<img src="./plate_1.jpg">

<img src="./plate_2.jpg">
