import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import torch.utils.tensorboard
import tensorboardX
from tensorboardX import SummaryWriter

"""根据宽度、深度、激活函数的不同，控制变量设置了八个网络，4层和6层网络，每个分两种宽度，每个网络设置两种激活函数，共八个网络"""


class NetRelu4Slim(nn.Module):
    def __init__(self):
        super(NetRelu4Slim, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 20), nn.ReLU(),
            nn.Linear(20, 50), nn.ReLU(),
            nn.Linear(50, 20), nn.ReLU(),
            nn.Linear(20, 1)
        )

    def forward(self, X):
        return self.net(X)


class NetRelu6Slim(nn.Module):
    def __init__(self):
        super(NetRelu6Slim, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 20), nn.ReLU(),
            nn.Linear(20, 50), nn.ReLU(),
            nn.Linear(50, 100), nn.ReLU(),
            nn.Linear(100, 50), nn.ReLU(),
            nn.Linear(50, 20), nn.ReLU(),
            nn.Linear(20, 1)
        )

    def forward(self, X):
        return self.net(X)


class NetRelu4Fat(nn.Module):
    def __init__(self):
        super(NetRelu4Fat, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 40), nn.ReLU(),
            nn.Linear(40, 100), nn.ReLU(),
            nn.Linear(100, 40), nn.ReLU(),
            nn.Linear(40, 1)
        )

    def forward(self, X):
        return self.net(X)


class NetRelu6Fat(nn.Module):
    def __init__(self):
        super(NetRelu6Fat, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 40), nn.ReLU(),
            nn.Linear(40, 100), nn.ReLU(),
            nn.Linear(100, 200), nn.ReLU(),
            nn.Linear(200, 100), nn.ReLU(),
            nn.Linear(100, 40), nn.ReLU(),
            nn.Linear(40, 1)
        )

    def forward(self, X):
        return self.net(X)


class NetSigmoid4Slim(nn.Module):
    def __init__(self):
        super(NetSigmoid4Slim, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 20), nn.Sigmoid(),
            nn.Linear(20, 50), nn.Sigmoid(),
            nn.Linear(50, 20), nn.Sigmoid(),
            nn.Linear(20, 1)
        )

    def forward(self, X):
        return self.net(X)


class NetSigmoid6Slim(nn.Module):
    def __init__(self):
        super(NetSigmoid6Slim, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 20), nn.Sigmoid(),
            nn.Linear(20, 50), nn.Sigmoid(),
            nn.Linear(50, 100), nn.Sigmoid(),
            nn.Linear(100, 50), nn.Sigmoid(),
            nn.Linear(50, 20), nn.Sigmoid(),
            nn.Linear(20, 1)
        )

    def forward(self, X):
        return self.net(X)


class NetSigmoid4Fat(nn.Module):
    def __init__(self):
        super(NetSigmoid4Fat, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 40), nn.Sigmoid(),
            nn.Linear(40, 100), nn.Sigmoid(),
            nn.Linear(100, 40), nn.Sigmoid(),
            nn.Linear(40, 1)
        )

    def forward(self, X):
        return self.net(X)


class NetSigmoid6Fat(nn.Module):
    def __init__(self):
        super(NetSigmoid6Fat, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 40), nn.Sigmoid(),
            nn.Linear(40, 100), nn.Sigmoid(),
            nn.Linear(100, 200), nn.Sigmoid(),
            nn.Linear(200, 100), nn.Sigmoid(),
            nn.Linear(100, 40), nn.Sigmoid(),
            nn.Linear(40, 1)
        )

    def forward(self, X):
        return self.net(X)


def data_load(num):
    """加载数据"""
    x = np.linspace(0, 4 * np.pi, num)
    y = np.sin(x)
    X = np.expand_dims(x, axis=1)
    Y = y.reshape(num, -1)
    return X, Y


def evaluate_accuracy(net_eva, test_iter, item_num):
    """评估测试集上的loss，采用均方误差"""
    if isinstance(net_eva, torch.nn.Module):
        net_eva.eval()
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    mse = 0
    for x, y in test_iter:
        x, y = x.to(device), y.to(device)
        y_hat_eva = net_eva(x)
        cmp = ((y_hat_eva - y) ** 2).sum()
        mse += cmp
    mse = mse / item_num
    return (mse ** 0.5).item()


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.weight, 0)
    # elif isinstance(m, nn.Conv2d):
        # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')


if __name__ == "__main__":
    writer = SummaryWriter('./path/to/log')
    lr, num_epochs, batch_size = 0.005, 600, 1000  #设置参数
    x_data, y_data = data_load(10000)

    db = torch.utils.data.TensorDataset(torch.tensor(x_data, dtype=torch.float),
                                        torch.tensor(y_data, dtype=torch.float))

    train_db, test_db = torch.utils.data.random_split(db, [8000, 2000])
    print(len(train_db), len(test_db))
    train_iter = torch.utils.data.DataLoader(train_db, batch_size=batch_size, shuffle=True)
    test_iter = torch.utils.data.DataLoader(test_db, batch_size=batch_size, shuffle=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(device)

    # 模型选择
    # net = NetRelu4Slim().to(device)
    # net = NetRelu6Slim().to(device)
    # net = NetRelu4Fat().to(device)
    # net = NetRelu6Fat().to(device)
    # net = NetSigmoid6Fat().to(device)
    # net = NetSigmoid4Fat().to(device)
    # net = NetSigmoid4Slim().to(device)
    net = NetSigmoid6Slim().to(device)


    net.apply(weight_init)
    loss = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    for epoch in range(num_epochs):
        net.train()
        l = None
        for x, y in train_iter:
            x, y = x.to(device), y.to(device)
            y_hat = net(x)
            l = loss(y, y_hat)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()

        print('epoch: ', epoch + 1, 'train loss: ', l.item())
        writer.add_scalar(net.__class__.__name__+' lr='+str(lr)+'/train_loss', l.item(), epoch + 1)
        with torch.no_grad():
            test_loss = evaluate_accuracy(net, test_iter, 2000)
            writer.add_scalar(net.__class__.__name__+' lr '+str(lr)+'/test_loss', l.item(), epoch + 1)
            print('epoch: ', epoch + 1, 'test loss: ', test_loss)

    net.cpu()
    net.eval()
    x = np.linspace(0, 4 * np.pi, 2000)
    y1 = np.sin(x)
    X = np.expand_dims(x, axis=1)
    Y = net(torch.tensor(X, dtype=torch.float))
    plt.plot(x, y1, label='y')
    plt.plot(x, Y.data.numpy(), label='y_hat')
    plt.title('sin func')
    plt.xlabel('x')
    plt.ylabel('sin(x)')
    plt.legend()
    plt.savefig(fname=net.__class__.__name__+'lr'+str(lr)+'.png')
    plt.show()
