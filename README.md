# 基于CNN的MNIST手写数据识别

## 文件描述

MNIST数据集是机器学习领域中非常经典的一个数据集，每个样本都是一张28 * 28像素的灰度手写数字图片。0~9十个手写数字。

Python版本数据集可以从网站http://yann.lecun.com/exdb/mnist/ 下载。一共包括四个文件，分别为训练集、训练集标签、测试集、测试集标签。若使用pytorch，也可以使用其自带的datasets库下载。

| 文件名称                   | 大小   | 内容                 |
| -------------------------- | ------ | -------------------- |
| train-images-idx3-ubyte.gz | 9681kb | 5000张验证集         |
| train-labels-idx1-ubyte.gz | 29kb   | 训练集图片对应的标签 |
| t10k-image-idx3-ubyte.gz   | 1611kb | 10000张测试集        |
| t10k-labels-idx-ubyte.gz   | 5kb    | 测试集图片对应的标签 |

本程序通过Pytorch自带的datasets库下载MNIST数据集，保存目录为`./data/MNIST/Raw/` ,并通过Pytorch的DataLoader库处理过，最后会使用位于`./data/MNIST/processed/` 目录下的文件作为训练集和测试集。

项目文件说明：

```
CNN_MNIST:.
│  data_loader.py  //训练集、测试集加载文件  
│  model.py  //CNN网络模型文件
│  sample_predict.py  //手写数字图片样例识别
│  test_sample.bmp  //手写数字图片样例
│  train.py  //模型训练
│
├─data
│  ├─MNIST
│  │  ├─processed
│  │  │      test.pt  //处理后的测试集
│  │  │      training.pt  //处理后的训练集
│  │  │      
│  │  └─raw  //处理前的MNIST数据
│  │          t10k-images-idx3-ubyte
│  │          t10k-images-idx3-ubyte.gz
│  │          t10k-labels-idx1-ubyte
│  │          t10k-labels-idx1-ubyte.gz
│  │          train-images-idx3-ubyte
│  │          train-images-idx3-ubyte.gz
│  │          train-labels-idx1-ubyte
│  │          train-labels-idx1-ubyte.gz
│  │          
│  └─test_samples  //更多的手写数字图片样例
│          1.bmp
│          10.bmp
│          2.bmp
│          3.bmp
│          4.bmp
│          5.bmp
│          6.bmp
│          7.bmp
│          8.bmp
│          9.bmp
│          
├─model_save  //模型保存
│      LeNet.pth
```



## 运行方法

### 运行环境

` python==3.7.7  pytorch==1.5.1  torchvision==0.6.1  numpy==1.18.1  pillow==7.1.2 `

`pip install packagename==version`



### 运行脚本

#### 1.训练与测试

`python train.py`

可选参数

```
--batch_size    default=64      每轮训练batch大小
--epochs        default=10      训练轮数
--LR            default=0.001   学习率
```

#### 2.使用图片进行预测

`python sample_predict.py`

可选参数

```
--filename      default=./test_sample.bmp      图片文件位置
--model         default=./model_save/LeNet.pth 加载模型位置
```

## 运行结果

`python train.py --epochs=10 --LR=0.001 --batch_size=64`

Test accuracy = 0.9887



`python sample_predict.py --filename=./test_sample.bmp --model=./model_save/LeNet.pth`

Tested picture ./test_sample.bmp

The hand writing number is: 2



![](./test_sample.bmp)
