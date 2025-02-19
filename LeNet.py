import share_function
import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np

#获取数据
train_iter , test_iter = share_function.data_iter()
batch_size = 64
dropout1 = 0.5
dropout2 = 0.2

#设计网络
net = nn.Sequential(
    nn.Conv2d(3, 32, kernel_size = 3, padding = 1), nn.ReLU(),
    nn.MaxPool2d(kernel_size = 2, stride = 2),
    nn.Conv2d(32, 64, kernel_size = 3, padding = 1), nn.ReLU(),
    nn.MaxPool2d(kernel_size = 2, stride = 2),
    nn.Flatten(),
    nn.Linear(64 * 8 * 8, 120), nn.ReLU(), nn.Dropout(dropout1),
    nn.Linear(120, 84), nn.ReLU(), nn.Dropout(dropout2),
    nn.Linear(84, 10)
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

num_epochs = 60
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

torch.save(net.state_dict(), 'model_Lenet.params')
