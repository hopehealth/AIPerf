# TF-Slim ImageNet数据集制作



TF-Slim(tensorflow.contrib.slim)是TensorFlow的高层API，类似layers，主要可以用来快速的设计、评估模型，有着类似keras般的语法  Tensorflow models包含一个Slim的图像分类库，可以微调、训练、使用预训练模型来对图像进行分类。

项目地址: https://github.com/tensorflow/models/tree/master/research/slim

### 数据集下载

官方提供四种数据集：  Flowers、CIFAR-10、MNIST、ImageNet-2012  前三个数据集数据量小，直接调用相关脚本自动会完成下载、转换（TFRecord格式）的过程，在  /userhome/AAH/scripts/build_data目录下执行以下脚本：

 官方下载地址：http://www.image-net.org/challenges/LSVRC/2012/nonpub-downloads ，需要“非.com结尾的邮箱注册的账号” 

```javascript
./download_imagenet.sh [dirname]
```

原始的ImageNet-2012下载下来包含三个文件:

- ILSVRC2012_bbox_train_v2.tar.gz
- ILSVRC2012_img_val.tar
- ILSVRC2012_img_train.tar

# ImageNet数据集制作

环境是Ubuntu 16.04 + CUDA10.1 + TF 2.2 + Python 3.5.

需要注意：  ImageNet数据集较大，下载慢

下载ImageNet-2012，转换(TFRecord).涉及如下三个脚本：

- preprocess_imagenet_validation_data.py`，处理val的数据
- process_bounding_boxes.py`，处理boundingbox数据
- build_imagenet_data.py`, 构建数据集主程序

#### 主程序`build_imagenet_data.py`:

训练集和验证集需要按照1000个子目录下包含图片的格式，处理步骤：

1. 将train 和 val 的数据按照文件夹分类
3. 指定参数运行build_imagenet_data.py

​	bbox就是bounding box数据（如无特殊需求，可不用，**本示例未使用bounding box**），另外两个是train 和 val 的数据（全是JPEG图片）

**可以按照以下步骤执行**:  假设数据存放在`models/research/slim/ImageNet-ori`目录下，TFRecord文件的输出目录是`models/research/slim/ILSVRC2012/output`，当前目录是`models/research/slim`

```shell
# 创建相关目录
mkdir -p ILSVRC2012  
mkdir -p ILSVRC2012/raw-data  
mkdir -p ILSVRC2012/raw-data/imagenet-data  

# 做验证集(解压时间久)
mkdir -p ILSVRC2012/raw-data/imagenet-data/validation/  
tar xf ILSVRC2012_img_val.tar -C ILSVRC2012/raw-data/imagenet-data/validation/
python preprocess_imagenet_validation_data.py ILSVRC2012/raw-data/imagenet-data/validation/ imagenet_2012_validation_synset_labels.txt

# 做训练集(解压时间更久，保持耐心!)
mkdir -p ILSVRC2012/raw-data/imagenet-data/train/
mv ImageNet-ori/ILSVRC2012_img_train.tar ILSVRC2012/raw-data/imagenet-data/train/ && cd ILSVRC2012/raw-data/imagenet-data/train/  
tar -xvf ILSVRC2012_img_train.tar && mv ILSVRC2012_img_train.tar ../../../ImageNet-ori/
find . -name "*.tar" | while read NAE ; do mkdir -p "${NAE%.tar}"; tar -xvf "${NAE}" -C "${NAE%.tar}"; rm -f "${NAE}"; done

# 执行准换
python build_imagenet_data.py --train_directory=ILSVRC2012/raw-data/imagenet-data/train --validation_directory=ILSVRC2012/raw-data/imagenet-data/validation --output_directory=ILSVRC2012/output --imagenet_metadata_file=imagenet_metadata.txt --labels_file=imagenet_lsvrc_2015_synsets.txt
```

上面步骤执行完后，路径models/research/slim/ILSVRC2012/output包含128个validation开头的验证集文件和1024个train开头的训练集文件。需要分别将验证集和数据集拷贝到共享目录/userhome/datasets下

```
mkdir /userhome/datasets/train
mkdir /userhome/datasets/val
cp  models/research/slim/ILSVRC2012/output/train-* /userhome/datasets/train
cp models/research/slim/ILSVRC2012/output/validation-* /userhome/datasets/val
```