"""
Author: LZF Zachary
Website: zrawberry.com
Filename: Utils.py
Function:
    加载数据集(生成或读取npy)
    格式化输入图片
"""
from os.path import exists

import numpy as np
import pandas as pd
from PIL import Image

# 数据集以及读取的npy存储路径
# origin_data = './Source/Dataset/small.csv'
origin_data = './Source/fer2013/fer2013.csv'
removed_data = './Source/fer2013/fer2013_rm.csv'
npy_save_path = './Source/Dataset/{}.npy'


def cls_remove():
    """
    # [0'愤怒', 1'恶心', 2'恐惧', 3'快乐', 4'悲伤', 5'惊讶', 6'平静'] -> [0'愤怒', 1'快乐', 2'悲伤', 3'惊讶', 4'平静']
    删除csv中的 恶心 恐惧表情
    :return:
    """
    print("从{0}读取原始数据集并删除部分类别".format(origin_data).center(60, '='))
    file = open(origin_data, 'r')
    df = pd.read_csv(file)
    file.close()

    df = df[~df['emotion'].isin([1, 2])]
    df.to_csv(removed_data)
    print("已将处理后的数据保存至{}".format(removed_data))


def load_data(mode='Train'):
    """
    根据模式从 csv文件或 npy文件中读取数据集
    :param mode: 加载数据的 模式  训练 开发 测试
    :return: images, labels
    """
    # mode_dict = {"Train": "Training", "Test": "PublicTest", "Dev": "PrivateTest"}
    mode_dict = {"Train": "Training", "Test": "PublicTest"}
    if mode not in mode_dict:
        print("未知的读取模式：" + mode)
        return None, None
    else:
        target = mode_dict[mode]

    if exists(npy_save_path.format('X_' + target)) and exists(npy_save_path.format('Y_' + target)):
        print("从{0}.npy中加载{0}数据集".format(target).center(60, '='))
        X = np.load(npy_save_path.format('X_' + target))
        Y = np.load(npy_save_path.format('Y_' + target))
        print("读取成功，数据大小 X:{} Y:{}".format(X.shape, Y.shape))
    else:
        print("从 \"{}\" 读取数据， 生成{}.npy".format(origin_data, target).center(60, '='))
        with open(removed_data, 'r') as file:
            csv = pd.read_csv(file)
            target_data = csv[csv['Usage'] == target]   # 选择目标的数据
            # target_data = csv[csv['Usage'] != 'PublicTest']  # 选择目标的数据 把privateTest也用作training 在生成test的时候注释
            X = []
            images_raw = target_data.loc[:, 'pixels'].tolist()
            for image_raw in images_raw:
                image = list(map(int, image_raw.split(' ')))
                image = np.asarray(image).reshape(48, 48)
                X.append(image)
            X = np.asarray(X)

            switch = {0: 0, 3: 1, 4: 2, 5: 3, 6: 4}
            Y_origin = target_data.loc[:, 'emotion'].tolist()
            Y = np.asarray([switch[cls] for cls in Y_origin if cls not in (1, 2)])
            # Y = np.asarray(target_data.loc[:, 'emotion'])
            np.save(npy_save_path.format('X_' + target), X)
            np.save(npy_save_path.format('Y_' + target), Y)
            print("生成完毕，数据大小 X:{} Y:{}".format(X.shape, Y.shape))
    return X, Y


def test_data(images, labels):
    EMOTIONS = ['愤怒', '快乐', '悲伤', '惊讶', '平静']

    print("展示数据集中的图片".center(60, '='))
    while True:
        try:
            index = int(input("输入图片索引号(0~{}):".format(len(labels))))
            label = labels[index]
            image = images[index]
            # 展示图片
            print("第{}号图片是{}表情".format(index, EMOTIONS[label]), end='')
            print(image.shape)
            image = Image.fromarray(image.astype('uint8'))
            image.show()
        except Exception as e:
            print(e)
            break


if __name__ == "__main__":
    # cls_remove()
    # load_data('Train')
    # load_data('Test')
    test_data(*load_data('Test'))
