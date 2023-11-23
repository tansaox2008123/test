import os
import re
import datetime
import time
import numpy as np
import random
import torch
import torch.distributed
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import argparse
import torch.nn.functional as F
import math
from multiprocessing import Pool
from sklearn.model_selection import train_test_split  # 将数据分为测试集和训练集
import os  # 打开读取文件
# import cv2  opencv读取图像
from matplotlib import pyplot as plt  # 测试图像是否读入进行绘制图像
import torch
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import ast
import torch


# 自定义排序键函数，将文件名解析为数字进行排序
def sort_key(filename):
    try:
        # 从文件名中提取数字部分
        number = int(''.join(filter(str.isdigit, filename)))
        return number
    except ValueError:
        # 如果文件名中没有数字部分，则默认返回 0
        return 0


def read_directory(directory_name, array_of_rna, length, rnaPath):
    # 参数为文件夹名称，rna数组，读取文件长度，rna地址数组
    direct = os.listdir(r"/home2/public/data/RNA_aptamer/sample_embedding/" + directory_name)
    # 乱序读进来的数据集，对读进来的数据集按照标号进行排序
    sorted_direct = sorted(direct, key=sort_key)
    # print(sorted_direct)
    if length == -1:
        length = len(direct)
    for filename in sorted_direct[0:length]:
        # 读取当前rna地址
        rPath = "/home2/public/data/RNA_aptamer/sample_embedding/" + directory_name + "/" + filename
        print(rPath)
        # 追加到rna地址数组中
        rnaPath.append(rPath)
        # 读取rna的npy文件
        # rna_embedding = np.load("/home2/public/data/RNA_aptamer/" + directory_name + "/" + filename)
        # 将（20，640）降维至一维12800
        # rna_embedding = np.array(rna_embedding).flatten()
        # array_of_rna.append(rna_embedding)


def read_tag(tag_path, tags, length):
    # 参数为标签文件路径，标签数组，读取长度
    # 读取数据的标签
    with open(tag_path) as file:
        for line in list(file)[0:length]:
            # 读取前多少个RNA的标签
            a = str(line).split()
            b = a[1]
            tag = test_rna(b)
            tags.append(tag)


def data_load(directory_name, tag_path, length, test_rate):
    # 获取数据与数据标签，并将数据分为训练集和测试集
    # 参数为rna文件夹，标签地址，length为读取的数据集长度

    # rna数组,标签数组，rna地址数组
    rnas = []
    tags = []
    rPath = []
    read_directory(directory_name, rnas, length, rPath)
    read_tag(tag_path, tags, length)

    rnas = np.array(rnas)
    tags = np.array(tags)

    return train_test_split(rnas, tags, test_size=test_rate)


def create_txt(directory_name, tag_path, length, test_rate):
    # rna数组
    rnas = []
    # rna 地址数组
    rna_path = []
    tags = []
    rPath = []
    read_directory(directory_name, rnas, length, rna_path)
    read_tag(tag_path, tags, length)

    x_train, x_test, y_train, y_test = train_test_split(rna_path, tags, test_size=test_rate)
    train = open('/home2/public/data/RNA_aptamer/data/sample_train2.txt', 'w+')
    test = open('/home2/public/data/RNA_aptamer/data/sample_test2.txt', 'w+')

    for i in range(len(x_train)):
        name = x_train[i] + ' ' + str(y_train[i]) + '\n'
        train.write(name)
    for i in range(len(x_test)):
        name = x_test[i] + ' ' + str(y_test[i]) + '\n'
        test.write(name)

    train.close()
    test.close()


encoding_dict = {'A': [1, 0, 0, 0],
                 'C': [0, 1, 0, 0],
                 'G': [0, 0, 1, 0],
                 'U': [0, 0, 0, 1]}

encoding_dict2 = {'A': [1],
                  'C': [2],
                  'G': [3],
                  'U': [4]}


# 对RNA进行编码
def test_rna(seq):
    # 你的RNA序列，长度为20
    print(seq)
    encoded_sequence = [encoding_dict2[base] for base in seq]
    flat_encoded_sequence = [item for sublist in encoded_sequence for item in sublist]
    return flat_encoded_sequence


def default_loader(path):
    return np.load(path)


def custom_transform(tensor):
    tensor = np.array(tensor).flatten()
    print(1)
    return tensor


class MyDataset(Dataset):

    def __init__(self, txt, transform=None, loader=default_loader):
        super(MyDataset, self).__init__()
        # 对继承自父类的属性进行初始化
        fh = open(txt, 'r')
        rnas = []
        # 按照传入的路径和txt文本参数，以只读的方式打开这个文本
        for line in fh:
            words = line.split()
            rnas.append((words[0], words[1]))
        self.rnas = rnas
        # 将（20，640）转化为12800
        # 使用 transforms.Compose 将自定义转换函数包括进来
        # 应用转换
        self.transform = transform
        # self.transform = transforms.Compose([transforms.Lambda(custom_transform)])
        self.loader = loader

    def __getitem__(self, index):
        fn1, seq = self.rnas[index]
        # fn是rna的path
        rna = self.loader(fn1)
        label = test_rna(seq)
        # label = self.loader2(fn2)

        # print(self.num)
        # self.num = self.num + 1q
        # 按照路径读取rna
        if self.transform is not None:
            print(2)
            rna = self.transform(rna)
        return rna, label

    def __len__(self):
        return len(self.rnas)


def MyDataLoader(directory_name, tag_path, length, test_rate, batch_size):
    # create_txt(directory_name, tag_path, length, test_rate)
    train_data = MyDataset(txt='/home2/public/data/RNA_aptamer/data/transformer_train_5.txt')
    test_data = MyDataset(txt='/home2/public/data/RNA_aptamer/data/transformer_test_5.txt')
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=False)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=False)
    return train_loader, test_loader


if __name__ == '__main__':
    # extract()
    # extract_shell()
    # probability()
    # find20()
    # softmax()
    # exchange_RNA()
    # fasta()
    # pre_data()
    # change_path()
    # test_rna()
    print(1)
    directory_name = 'representations'
    tag_txt = 'sample_pro.txt'
    tag_path = '/home2/public/data/RNA_aptamer/RNA-pro/' + tag_txt
    length = 24181
    test_rate = 0.2
    # create_txt(directory_name, tag_path, length, test_rate)
    file_path = '/home2/public/data/RNA_aptamer/sample_embedding/representations/2762.npy'

    loaded_data = np.load(file_path)
    print("Loaded Data Shape:", loaded_data.shape)
    print(loaded_data)
