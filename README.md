# Robomaster装甲板数字识别

### 项目介绍

Robomaster2020赛季装甲板数字识别，包括Python（PyTorch框架）训练模型，测试模型，导出模型，在C++上（LibTorch）配合OpenCV部署模型



### 目录结构说明

```
.
├── C++模型部署                          模型部署
│   ├── CMakeLists.txt					
│   ├── image                           测试图片
│   ├── main.cpp                        主程序
│   └── model                           模型参数
├── learn                               模型训练
│   ├── __init__.py
│   └── mymodel.py
├── libtorch_model                      供LibTorch使用的模型参数
│   └── model.pt                        网络结构+具体参数
├── LICENSE
├── model_param                         供模型测试使用的模型参数
│   └── state_dict.pt                   仅具体含参数
├── README.md
├── test                                模型测试
│   ├── result.csv                      测试结果
│   └── test.py                         测试程序
├── tool
│   └── make_label.py                   制作数据标签
└── 素材
    └── num                             装甲板数字数据集

```

### 系统环境

- Ubuntu 18.04
- PyTorch/LibToch 1.4.0 CUDA版本
- CUDA 10.1
- cuDNN 7.6.5
- CMake 3.10.2
- OpenCV 3.4.3
- Python 3.7.6


### 模型训练说明

进入项目目录

#### 制作标签
```
python tool/make_label.py
```

#### 训练模型
```
python learn/mymodel.py
```

#### 测试模型
```
python test/test.py
```

#### 部署模型
```
cd C++模型部署/
mkdir build
cd build
cmake ..
make
./deploy
```

### 其他
更详细的说明参见我的语雀文档：https://www.yuque.com/herormluzhan/rghypl/ihpfmk
