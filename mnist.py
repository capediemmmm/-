import torch
import numpy as np
import matplotlib.pyplot as plt

# 一、Tensor(张量)是pytorch中的特殊数据结构：

# 初始化一个张量：
# 1.张量可以直接从一类data中获取，data的类型会被自动检测：

data  = [[1, 2], [3, 4]]
x_data = torch.tensor(data)

# 2.张量可以从numpy的array中创造出来：
np_array = np.array(data)
x_np = torch.from_numpy(np_array)

# 3.张量可以从另一个张量获得：

x_ones = torch.ones_like(x_data)  # 保留x的数据特性
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float)  # 重新定义x的属性
print(f"Random Tensor: \n {x_rand} \n")

# tuple类型的数据会影响函数的输出数据格式，即决定了生成的数据结构：

import torch

shape = (2,4,)  # 二行四列
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)  # 生成一个元素值全为1的张量
zeros_tensor = torch.zeros(shape)  

print(f"Rand Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor} \n") 

import torch

tensor = torch.rand(3, 4) # 三行四列

print(f"Shape of Tensor: {tensor.shape} \n")  # torch.size([3, 4])
print(f"DataType of Tensor: {tensor.dtype} \n")
print(f"Device Tensor is stored in: {tensor.device} \n")
# storage直接返回一个指向向量内部基础存储的指针，这样可以直接操作张量的内部存储，从而绕开pytorch
import torch 

x = torch.tensor([[1, 2], [3, 4]])
storage = x.storage()
storage[0] = 0;
print(x)

# 有关tensor的操作： https://pytorch.org/docs/stable/torch.html
import torch
# tensor的检索和滑动很像numpy中的array
tensor = torch.ones(4, 4)
print(f"First Row:  {tensor[0]} ")
print(f"First Column: {tensor[:, 0]}")
print(f"Last Column: {tensor[:, -1]}")
tensor[:, 1] = 0
print(tensor)

tensor1 = torch.ones(4, 4)
t1 = torch.cat([tensor, tensor1, tensor1], dim =-1 )  # torch.cat中的dim range是[-2,1]分别表示下、右、上、左
print(t1)


# 数学操作：
# Pytorch中 @ 表示矩阵乘法运算

import torch
tensor = torch.ones(4, 4)

y1 = tensor @ tensor.T  # tensor.T表示矩阵的转置
y2 = tensor.matmul(tensor.T) # .matmul()同样表示矩阵乘法

y3 = torch.rand_like(y1)
torch.matmul(tensor, tensor.T, out=y3) # 乘法结果输出到y3

print(f" {y1} \n {y2} \n {y3} \n")

# * 表示元素级乘法操作，也就是将两个张量对应的元素逐个相乘，其中两个张量形状必须相同，而矩阵乘法@要求的是两个张量维度匹配
z1 = tensor * tensor
z2 = tensor.mul(tensor)

z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)
print(f" {z1} \n {z2} \n {z3}")

# 使用.item()可以将一个只有一个元素的张量转换成python中相对应的数据类型 （Note:）In-place operations save some memory, but can be problematic when computing derivatives because of an immediate loss of history. Hence, their use is discouraged.
import torch 
tensor = torch.ones(4, 4)
agg = tensor.sum()  # 此时agg仍是一个张量
agg_item = agg.item()
print(agg_item, type(agg_item))

# in-place operations : 将改变存到操作对象上的操作称为in-place，这样的操作一般用_来标志：
import torch
tensor = torch.ones(4, 4)
print(f"{tensor} \n")
tensor.add_(5)  
print(tensor)

# pytorch和numpy的联系：CPU上的张量和numpy的arrays可以共享他们的地址，即改变其中一个，另一个也会改变
# 1.tensor to numpy arrays:
import torch
t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")
t.add_(1)
print(f"t: {t}")
print(f"n: {n}")

# 2.numpy arrays to tensor:
import torch
import numpy as np

n = np.ones(5)
t = torch.from_numpy(n)
print(f"n: {n}")
print(f"t: {t}")

np.add(n, 1, out=n)
print(f"n: {n}")
print(f"t: {t}")

# 二、DATASETS & DATALOADERS：pytorch提供了两个数据库：torch.utils.data.DataLoader和torch.utils.data.Dataset
# 两者区别：DATASETS是一个抽象类，代表一个数据集，而DATALOADERS则是一个数据加载器，它可以加载一个DATASETS对象并生成一个迭代器，用于对数据进行批次处理
# 1.Loading a dataset:
# Here is an example of how to load the Fashion-MNIST dataset from TorchVision. Fashion-MNIST is a dataset of Zalando’s article images consisting of 60,000 training examples and 10,000 test examples. Each example comprises a 28×28 grayscale image and an associated label from one of 10 classes.
# 我们加载FashionMnist Dataset的如下参数：
# root: train/test数据存储的路径
# train: 区分train和test dataset
# download=True: 如果root的path失效的话从网络下载  
# transform 和 target_transform：用于指定如何对数据进行预处理

import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor # 用于将数据转换为张量形式
import matplotlib.pyplot as plt

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)
test_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)
# 下载后再执行代码不会重复下载

# Iterating and Visualizing the Dataset:

labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}
figure = plt.figure(figsize = (8, 8))
rows, cols = 3, 3
for i in range(1, rows*cols+1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off") # 隐藏绘图中的坐标轴
    plt.imshow(img.squeeze(), cmap="gray")  #.squeeze()是将张量中维数大小为1的维度移除，作为一种压缩张量的手段，使得张量更方便处理，plt.imshow()是将img输出到matplot上
plt.show()  # 将matplot中绘制的图像输出到屏幕上

# Creating a Custom Dataset(自定义数据集) for your files:
# A custom Dataset class must implement three functions: __init__, __len__, and __getitem__.
import os
import pandas as pd
from torchvision.io import read_image  # read_image用于通过一段路径打开一个图片

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)   # pandas.read_csv()函数用于读取一个csv(逗号分隔值)文件并将其转换为pandas的dataframe对象
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0]) # .iloc()是pandas中的一个按位置索引选择数据的函数，可以选择一行、一列或者是多行、多列的数据
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

# Preparing your data for training with DataLoaders:
# 在机器学习中，数据集是用于训练模型的样本集合。每个样本通常包含一个或多个特征（输入）和一个或多个标签（输出）。在训练期间，模型学习根据样本的特征进行预测。
# 在训练模型时，通常希望将样本分成“小批量”而不是逐个样本传递。小批量是数据集的一个小子集，通常包含16、32或64个样本。通过将样本分成小批量传递，我们可以利用现代硬件中的矢量化操作，从而显著加快训练过程。
# 为了防止模型过度拟合，我们还希望在每个时期随机重新排列数据。这意味着在每次训练时，我们会随机重新排列数据集中的样本。重新排列有助于确保模型不会学习依赖样本顺序中的特定模式。
# 最后，为了加速数据检索，我们可以使用Python的多进程模块并行加载样本。这意味着我们可以同时加载多个样本，而不是逐个加载样本，这可以显著减少加载整个数据集所需的时间。
# 在PyTorch中，Dataset类提供了一个接口，用于检索数据集中样本的特征和标签。DataLoader类可以用于迭代样本的小批量，并自动在每个时期重新排列数据。通过设置DataLoader构造函数的num_workers参数，我们还可以使用Python的多进程模块加速数据检索。
from torch.utils.data import DataLoader
train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True) # DataLoader类返回值是样本张量和标签(也是一个张量)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

# 创建DataLoader对象，用于迭代训练集中的样本，shuflle决定是否在每个时期对样本重排，batch_size参数决定了每个小批量包含的样本数，在每次迭代中，DataLoader对象会自动从训练数据集中取出一个小批量的样本，并将其转换成一个张量（tensor）对象，可以直接用于训练模型。

# Iterate through the DataLoader：
# 将dataset中的数据加载为dataloader对象后，就可以按需要对其进行迭代，下面的程序中，每次迭代返回一个小批量的train_features和train_labels
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")

# Transforms:
# 不是所有的数据下载下来都是能够用于机器学习的类型，因此我们需要通过transform来将其转换为我们需要的类型，所有的torchvision dataset都有两个参数，分别是transform--特征相关，target_transform--标签相关
# torchvision.transforms提供了一些常用的转换操作
# FashionMNIST中的数据特征和PIL的图像格式是一样的，labels是整数，为了训练，我们需要将向量归一化，将标签转为one-hot向量表示（方便计算）
# one-hot编码（one-hot encoding）是一种常用的向量表示方法。对于一个具有n个类别的分类问题，将每个类别编码为一个长度为n的向量，其中只有对应类别的位置上的值为1，其他位置上的值均为0。这样的向量称为one-hot向量。
# 我们用ToTensor和lambda来完成以上操作
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

ds = datasets.FashionMNIST(
    root = "data1",
    train = True,
    download = True,
    transform = ToTensor(),
    target_transform = Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)) # 用于将y转换为one-hot向量，这里新建一个长度为10的零向量，使用scatter函数在零向量上将y对应的值设为1


)
