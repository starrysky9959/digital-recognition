"""
@author  starrysky
@date    2020/08/16
@details 加载训练好的模型参数, 测试模型, 导出完整的模型供C++项目部署使用
"""

import sys

sys.path.append("./")

import time
import torch
from learn import mymodel
import os
import pandas as pd

# %%
# print(os.getcwd())
# os.chdir(os.getcwd() + r"\test")
print(os.getcwd())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
model = mymodel.LeNet()
model.load_state_dict(torch.load("./model_param/state_dict.pt"))
model.eval()
print(model)

# 加载样本和标签
df = pd.read_csv("./素材/num/label.csv", index_col=0)
df["predict"] = None
df["is_correct"] = None
print(df.columns)

# %%
ans = 0
start_time = time.time()
for i in range(df.shape[0]):
    image_path = df.iloc[i, 0]
    image = mymodel.default_loader(image_path)
    label = df.iloc[i, 1]
    x = mymodel.trans(image)
    x_ = x.view(1, 1, 28, 28)
    y_predict = model(x_).argmax(dim=1).item()
    # print(i, y_predict)
    df.iloc[i, 2] = y_predict
    df.iloc[i, 3] = y_predict == label

    if y_predict == label:
        ans += 1

print("正确样本数:{0}, 正确率={1:.4f}".format(ans, ans / df.shape[0]))
print("测试时间:{0:.4f}".format(time.time() - start_time))

# %%
df.to_csv("result.csv", index=False)
image_path = df.iloc[0, 0]
image = mymodel.default_loader(image_path)
x = mymodel.trans(image)
x_ = x.view(1, 1, 28, 28)
traced_script_module = torch.jit.trace(model, x_)
traced_script_module.save("./libtorch_model/model.pt")
