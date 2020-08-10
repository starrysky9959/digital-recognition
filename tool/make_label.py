import pandas as pd
import os

df = pd.DataFrame({
    "image_path": [],
    "label": [],
})

# 素材文件目录路径
src_dir = r"../素材/num/0-processed/0/"
files = os.listdir(src_dir)
for i in files:
    # print(src_dir + i)
    df.loc[df.shape[0] + 1] = {
        "image_path": src_dir + i,
        "label": 0,
    }

dir_type_list = ["close", "closeh", "far", "farh"]
label_type_list = ["num1", "num2", "num3", "num4", "num5", "sentinel"]
for num in range(0, 6):
    for dir_name in dir_type_list:
        # 素材文件目录路径
        src_dir = r"../素材/num/" + label_type_list[num] + "-processed/" + label_type_list[num] + "/" + dir_name + "/"
        files = os.listdir(src_dir)
        for i in files:
            # print(src_dir + i)
            df.loc[df.shape[0] + 1] = {
                "image_path": src_dir + i,
                "label": num + 1,
            }

# print(df)
df.to_csv("../素材/num/label.csv")
