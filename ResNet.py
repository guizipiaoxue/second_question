import share_function
import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
from torch.nn import functional as F

#获取数据
train_iter , test_iter = share_function.data_iter()
batch_size = 64
dropout1 = 0.5
dropout2 = 0.2

class Residual(nn.Module):
    def __init__(self, in_channels, out_channels,
                 use_1x1conv = False , stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3 is not None:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)

b1 = nn.Sequential(
    nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(64), nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2)
)

def resnet_block(input_channels, out_channels, num_residual, first_block=False):
    blk = []
    for i in range(num_residual):
        if i == 0 and not first_block:
            blk.append(Residual(in_channels=input_channels,
                                out_channels=out_channels,use_1x1conv=True))
        else:
            blk.append(Residual(in_channels=out_channels, out_channels=out_channels))
    return blk

b2 = nn.Sequential(*resnet_block(input_channels=64, out_channels=64, num_residual=2, first_block=True))
b3 = nn.Sequential(*resnet_block(input_channels=64, out_channels=128, num_residual=2))
b4 = nn.Sequential(*resnet_block(input_channels=128, out_channels=128, num_residual=2))
b5 = nn.Sequential(*resnet_block(input_channels=128, out_channels=256, num_residual=2))
b6 = nn.Sequential(*resnet_block(input_channels=256, out_channels=256, num_residual=2))
b7 = nn.Sequential(*resnet_block(input_channels=256, out_channels=512, num_residual=2))
b8 = nn.Sequential(*resnet_block(input_channels=512, out_channels=512, num_residual=2))

net = nn.Sequential(b1, b2, b3, b4, b5, b6, b7, b8,
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(in_features= 512, out_features=10),
)

def train(net , train_iter, test_iter, num_epochs, lr, device = 'cuda:0'):
    #定义参数
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr = lr)
    loss = nn.CrossEntropyLoss()
    train_loss = []
    train_acc = []
    test_acc = []
    for epoch in range(num_epochs):
        net.train()
        total_loss, total_correct, total_samples = 0.0, 0, 0
        for i ,(X, y) in enumerate(train_iter):
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            total_loss += l.item() * X.shape[0]
            total_correct += share_function.acc(y_hat, y)
            total_samples += X.shape[0]

        train_loss.append(total_loss / total_samples)
        train_acc.append(total_correct / total_samples)

        net.eval()
        test_correct, test_total = 0, 0
        for i, (X, y) in enumerate(test_iter):
            X, y = X.to(device), y.to(device)
            test_correct += share_function.acc(net(X), y)
            test_total += X.shape[0]
        test_acc.append(test_correct / test_total)
    return train_loss, train_acc, test_acc

num_epochs = 50
lr = 0.01
device = 'cuda:0'
train_loss , train_acc, test_acc = train(net, train_iter, test_iter, num_epochs, lr, device)
plt.plot(np.linspace(0, num_epochs, len(train_loss)), train_loss, label = 'train_loss', color = 'blue')
plt.plot(np.linspace(0, num_epochs, len(train_acc)), train_acc, label = 'train_acc', color = 'green')
plt.plot(np.linspace(0, num_epochs, len(test_acc)), test_acc, label = 'test_acc', color = 'red')
plt.title(f'train_acc:{train_acc[-1]}  test_acc:{test_acc[-1]}')
plt.legend()
plt.xlabel('epoch')
plt.show()

print('train_acc',train_acc[-1])
print('test_acc',test_acc[-1])
print('train_loss',train_loss[-1])

torch.save(net.state_dict(), 'model_resnet.params')