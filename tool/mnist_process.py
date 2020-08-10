"""
用于处理mnist数据集
"""

import os
from skimage import io
import torchvision.datasets.mnist as mnist

root = "/home/luzhan/My-Project/Python/PyTorch/数字识别/data/"

train_set = (
    mnist.read_image_file(os.path.join(root, 'train-images.idx3-ubyte')),
    mnist.read_label_file(os.path.join(root, 'train-labels.idx1-ubyte'))
)

test_set = (
    mnist.read_image_file(os.path.join(root, 't10k-images.idx3-ubyte')),
    mnist.read_label_file(os.path.join(root, 't10k-labels.idx1-ubyte'))
)

print("train set:", train_set[0].size())
print("test set:", test_set[0].size())


def convert_to_img(train=True):
    if (train):
        f = open(root + 'train.txt', 'w')
        data_path = root + '/train/'
        if (not os.path.exists(data_path)):
            os.makedirs(data_path)
        for i, (img, label) in enumerate(zip(train_set[0], train_set[1])):
            img_path = data_path + str(i) + '.jpg'
            io.imsave(img_path, img.numpy())
            f.write(data_path + str(i) + '.jpg ' + str(1) + '\n')
        for i in range(60000, 80000):
            f.write(data_path + str(i) + '.jpg ' + str(0) + '\n')

        f.close()
    else:
        f = open(root + 'test.txt', 'w')
        data_path = root + '/test/'
        if (not os.path.exists(data_path)):
            os.makedirs(data_path)
        for i, (img, label) in enumerate(zip(test_set[0], test_set[1])):
            img_path = data_path + str(i) + '.jpg'
            io.imsave(img_path, img.numpy())
            f.write(data_path + str(i) + '.jpg ' + str(1) + '\n')
        for i in range(10000, 12000):
            f.write(data_path + str(i) + '.jpg ' + str(0) + '\n')
        f.close()


convert_to_img(True)
convert_to_img(False)
