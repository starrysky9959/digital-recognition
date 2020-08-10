# Robomaster装甲板数字识别

### 介绍

Robomaster2020赛季装甲板数字识别，Python（Pytorch框架）训练模型，导出模型，C++（LibTorch，PyTorch的C++接口）部署模型



### 项目结构说明

```
├─learn							模型定义和训练
├─libtorch_model				供C++ LibTorch 使用的完整的模型参数
├─model_param					网络参数（不含网络结构）
├─test							模型测试并导出
├─tool							打标签
└─素材							数据集
```



