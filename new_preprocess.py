'''
重新写一个可以根据自己设定的数来提取训练集的数据处理程序
'''

import os
import random

import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.io as sio
import argparse
import collections
from sklearn.model_selection import train_test_split

def seed_torch(seed=1029):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

seed_torch()



parser = argparse.ArgumentParser("PU")

parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--num_of_segment', type=int, default=6, help='Divide the data processing process into several stages')
parser.add_argument('--windows', type=int, default=5, help='patch size')
parser.add_argument('--sample', type=int, default=200, help='sample sizes for training')
parser.add_argument('--simple_percent', type=int, default=0.7, help='sample sizes for training')

args = parser.parse_args()

#传入选中的像素的index，pad_length，以及原始数据的H,W
def indexToAssignment(index_, pad_length, Row, Col):
    new_assign = {}
    for counter, value in enumerate(index_):
        #要加上pad_length是因为之前对整体数据进行了二维padding
        #别管为什么是这样的，反正我试了一下是对的
        assign_0 = value // Col + pad_length #X坐标轴
        assign_1 = value % Col + pad_length #y坐标轴
        new_assign[counter] = [assign_0, assign_1]
    return new_assign


def assignmentToIndex(assign_0, assign_1, Row, Col):
    new_index = assign_0 * Col + assign_1
    return new_index


def selectNeighboringPatch(matrix, ex_len, pos_row, pos_col):
    # print(matrix.shape)
    #说白了就是先选中行，再选中列，没毛病
    selected_rows = matrix[:, range(pos_row - ex_len, pos_row + ex_len + 1), :]
    selected_patch = selected_rows[:, :, range(pos_col - ex_len, pos_col + ex_len + 1)]
    return selected_patch

#这里只能是一个数据集一个数据集地来进行设定了，
#总共有267z
#42776
if args.simple_percent == 1:
    sample_200 = [66, 200, 22, 32, 16, 55, 26, 32, 10]  #1%
elif args.simple_percent == 0.5:
    sample_200 = [33, 100, 11, 16, 8, 26, 13, 16, 5]
elif args.simple_percent == 3:
    sample_200 = [198, 600, 66, 96, 48, 165, 78, 96, 30]
elif args.simple_percent == 2:
    sample_200 = [132, 400, 44, 64, 32, 110, 52, 64, 20]
elif args.simple_percent == 0.3:
    sample_200 = [20, 56, 6, 9, 4, 15, 4, 11, 3]
elif args.simple_percent == 0.1:
    sample_200 = [6, 19, 2, 3, 2, 5, 2, 4, 2]
elif args.simple_percent == 0.7:
    sample_200 = [42, 133, 14, 21, 14, 35, 14, 28, 14]



sample_400 = [178, 20, 9, 9, 17, 24, 19, 115, 9]

sample_200 = [2 * i for i in sample_200]

if args.sample == 200:
    SAMPLE = sample_200
elif args.sample == 400:
    SAMPLE = sample_400
else:
    print('Sample size is out of range.')


#分割训练集和测试集
def rSampling(groundTruth, sample_num=SAMPLE):  # divide datasets into train and test datasets
    #保存了每一个类中像素点对应的坐标位置
    whole_loc = {}
    train = {}
    #val实际上是test，但是在搜索的过程中确实是val
    val = {}
    m = np.max(groundTruth)
    #按照每一个类别进行循环取出相应的坐标
    for i in range(m):
        #从第一个类开始，把所有属于第一个类的坐标全部提取出来
        #这里有个很细节的地方，直接x == i + 1就可以把未标记的像素点给去除了
        indices = [j for j, x in enumerate(groundTruth.ravel().tolist()) if x == i + 1]
        #这个好，随机洗牌就打消了我们从整体数据集中取除训练集的顾虑
        np.random.shuffle(indices)
        #保存到loc的对应位置
        whole_loc[i] = indices
        #随机以后取前一部分作为训练集，后一部分作为测试集
        train[i] = indices[:sample_num[i]]
        val[i] = indices[sample_num[i]:]

    #当所有的坐标已经获取，将字典里的索引依次append到列表中
    #看似这里的whole_indices没什么用，但是实际上
    whole_indices = []
    train_indices = []
    val_indices = []
    for i in range(m):
        whole_indices += whole_loc[i]
        train_indices += train[i]
        val_indices += val[i]
        #这两操作我觉得可以在循环外面来做
        np.random.shuffle(train_indices)
        np.random.shuffle(val_indices)
    return whole_indices, train_indices, val_indices

#就是在数据的周围打上一圈0，channel这个维度上不打0
def zeroPadding_3D(old_matrix, pad_length, pad_depth=0):
    #由于输入的是一个3D数据，所以需要pad_depth这个值
    new_matrix = np.lib.pad(old_matrix, ((pad_depth, pad_depth), (pad_length, pad_length), (pad_length, pad_length)),
                            'constant', constant_values=0)
    return new_matrix


#这个函数仅仅实现按照训练集或者测试集的长度将trainset或者testset每段分割的长度以列表形式给出即可
#data:list n:int
#少了一段，等下回来找一下是哪里的问题
def split_indices(data,n):
    #print("分割长度")
    total_length = len(data)
    #print("total_length:{}".format(total_length))
    each_length = total_length//n

    splited_length = []
    for i in range(n):
        #除最后一段可能会出现除不尽的情况要单独处理之外，其他的统一一个循环即可
        if i !=n-1:
            splited_length.append(each_length)
        elif i == n-1:
            splited_length.append(total_length-each_length*(i))

    return splited_length

#统计训练集中每一个类的数量
def countEachClassInTrain(y_count_train,num_class):
    each_class_num=np.zeros([num_class])
    for i in y_count_train:
        i=int(i)
        each_class_num[i]=each_class_num[i]+1
    return each_class_num

mat_data = sio.loadmat('PaviaU.mat')
data_IN = mat_data['paviaU']
mat_gt = sio.loadmat('PaviaU_gt.mat')
gt_IN = mat_gt['Data_gt']

print(data_IN.shape)

bands = data_IN.shape[-1]
nb_classes = np.max(gt_IN)

#其他的改好了，开始看吧
# Input datasets configuration to generate 102x9x9 HSI samples
new_gt_IN = gt_IN

# img_rows, img_cols =  7, 7 # 9, 9

#这两啥意思没搞懂，后面应该可以简化，到时候再看
INPUT_DIMENSION_CONV = bands
INPUT_DIMENSION = bands


# 总的样本数，（去除未标记样本以后的数）
TOTAL_SIZE = np.sum(gt_IN>0)

#训练集的大小，
TRAIN_SIZE = sum(SAMPLE)
print("")

#测试集的大小就是总的数量减去训练集的大小
TEST_SIZE = TOTAL_SIZE - TRAIN_SIZE

img_channels = bands
#patches每一边的长度
PATCH_LENGTH = int(args.windows/2)  # Patch_size 9*9

#取data数据中的最大值
MAX = data_IN.max()
#将(H,W,C)数据转换为(C,H,W)
data_IN = np.transpose(data_IN, (2, 0, 1))

#这里做了归一化处理
data_IN = data_IN - np.mean(data_IN, axis=(1, 2), keepdims=True)
data_IN = data_IN / MAX

#将数据转换为(C,H*W),非得那么费劲
data = data_IN.reshape(np.prod(data_IN.shape[:1]), np.prod(data_IN.shape[1:]))

#将标签转换为(H*W, )的形式
gt = new_gt_IN.reshape(np.prod(new_gt_IN.shape[:2]), )

#就是data_IN,不明白为什么要重新创建一个
whole_data = data.reshape(data_IN.shape[0], data_IN.shape[1], data_IN.shape[2])

#看下为什么数据出来会是(200,159,159),这个数据是对的，一遍加了7，两边不久总共加了15嘛
padded_data = zeroPadding_3D(whole_data, PATCH_LENGTH)

CATEGORY = nb_classes

#返回对应的：所有可用像素的坐标，train的坐标，test的坐标 的列表
all_indices, train_indices, test_indices = rSampling(gt)

#根据坐标取到对应的标签值， 为什么要减1倒是没搞懂,可能是argmax出来的第一个类的分类是0
#这会不会就是SSTN出来分类性能极差的原因
y_train = gt[train_indices] - 1
y_test = gt[test_indices] - 1
print('训练集长度(按照label核算)：{}'.format(len(y_train)))
print('测试集长度(按照label核算)：{}'.format(len(y_test)))

#从这里要将数据进行分段处理，看似train和test要一起进行分段处理，实则不然，两者可以分开进行分段处理
#1、按照各自的长度，分成相应的段,因为后面是一个patch一个patch提取的，这里只需要返回各段的长度即可，无需真正将数据进行分段
print('训练集长度(按照data核算)：{}'.format(len(train_indices)))
print('测试集长度(按照data核算)：{}'.format(len(test_indices)))
splited_train_len = split_indices(train_indices,args.num_of_segment)
print("训练集分段情况：")
print(splited_train_len)
splited_test_len = split_indices(test_indices,args.num_of_segment)
print("测试集分段情况：")
print(splited_test_len)

#创建一个(0,C,15,15)的容器,最后处理的结果全部concat到里面来就行
X_train=np.empty([0,bands,args.windows,args.windows])
X_test=np.empty([0,bands,args.windows,args.windows])

#正式开始进行patches提取的大循环
for i in range(args.num_of_segment):
    print("---------分割第{}段---------".format(i + 1))
    X_train_i = np.empty([splited_train_len[i], bands, args.windows, args.windows])
    X_test_i = np.empty([splited_test_len[i], bands, args.windows, args.windows])
    #whole_data:(C,H,W)就是所有数据
    #取H和W送入函数，得到选中像素在原来H*W坐标系下的二维坐标(字典)
    #麻烦的事情是要分段送入坐标
    print("start_index:{}".format(i*splited_train_len[1]))
    print("end_index:{}".format(i*splited_train_len[1]+splited_train_len[i]-1))
    #将在H*W中的index转换为(H*W)中的index
    train_assign = indexToAssignment(train_indices[i*splited_train_len[1]:i*splited_train_len[1]+splited_train_len[i]-1],
                                     PATCH_LENGTH, whole_data.shape[1], whole_data.shape[2])

    #对选中的像素逐一生成patch送入到创建好的零矩阵中
    for j in range(len(train_assign)):
        X_train_i[j] = selectNeighboringPatch(padded_data, PATCH_LENGTH, train_assign[j][0], train_assign[j][1])

    test_assign = indexToAssignment(test_indices[i*splited_test_len[1]:i*splited_test_len[1]+splited_test_len[i]-1],
                                    PATCH_LENGTH, whole_data.shape[1], whole_data.shape[2])
    for j in range(len(test_assign)):
        X_test_i[j] = selectNeighboringPatch(padded_data, PATCH_LENGTH, test_assign[j][0], test_assign[j][1])

    X_train=np.vstack((X_train,X_train_i))
    X_test=np.vstack((X_test,X_test_i))

#当我们有stratify=y，就代表分割的时候将各类按比例选取
def splitTrainValSet(X, y, testRatio=0.50):
    print("分割前y.size={}".format(y.size))
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=testRatio, random_state=345,
                                                        stratify=y)
    print("训练集的大小：{}".format(y_train.size))
    print("val集的大小:{}".format(y_val.size))
    return X_train, X_val, y_train, y_val

def savePreprocessedData(X_trainPatches, X_valPatches, X_testPatches, y_trainPatches, y_valPatches, y_testPatches,
                        windowSize):
    with open(str(args.simple_percent) + "%XtrainWindowSize" + str(windowSize) + ".npy", 'bw') as outfile:
        np.save(outfile, X_trainPatches)
    with open(str(args.simple_percent) + "%XvalWindowSize" + str(windowSize) + ".npy", 'bw') as outfile:
        np.save(outfile, X_valPatches)
    with open(str(args.simple_percent) + "%XtestWindowSize" + str(windowSize) + ".npy", 'bw') as outfile:
        np.save(outfile, X_testPatches)
    with open(str(args.simple_percent) + "%ytrainWindowSize" + str(windowSize) + ".npy", 'bw') as outfile:
        np.save(outfile, y_trainPatches)
    with open(str(args.simple_percent) + "%yvalWindowSize" + str(windowSize) + ".npy", 'bw') as outfile:
        np.save(outfile, y_valPatches)
    with open(str(args.simple_percent) + "%ytestWindowSize" + str(windowSize) + ".npy", 'bw') as outfile:
        np.save(outfile, y_testPatches)

#提取完了以后将训练集分离为test和val
X_train,X_val,y_train,y_val = splitTrainValSet(X_train,y_train,testRatio=0.5)

#统计一下各个类别的数量，核验一下
print("train集各类别的数量：")
print(countEachClassInTrain(y_train,nb_classes))
print("val集各类别的数量：")
print(countEachClassInTrain(y_val,nb_classes))
print("test集各类别的数量：")
print(countEachClassInTrain(y_test,nb_classes))

savePreprocessedData(X_train, X_val, X_test, y_train, y_val, y_test, windowSize=args.windows)