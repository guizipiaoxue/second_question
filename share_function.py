import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils import data

trans = trans_train = transforms.Compose([
    transforms.ToTensor()])
train_data = datasets.CIFAR10(root="../data", train=True, transform=trans, download=True) # 下载数据集
train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True) # 打乱并分批次读取数据
test_data = datasets.CIFAR10(root="../data", train=False, transform=trans, download= True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=False)

def data_iter():
    return train_loader, test_loader

def acc(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = torch.argmax(y_hat, dim=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())
