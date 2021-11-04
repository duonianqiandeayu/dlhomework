import os
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
# import torch.utils.tensorboard
# import tensorboardX
# from tensorboardX import SummaryWriter
from PIL import Image
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import torch.nn.functional as F
import time

class Residual(nn.Module):
    def __init__(self, input_channels, num_channels, use1x1conv=False, strides=1):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        if use1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y= self.bn2(self.conv2(Y))
        if self.conv3:
            X= self.conv3(X)
        Y+=X
        return F.relu(Y)

class ResNet34(nn.Module):
    def __init__(self, num_classes=100):
        super(ResNet34, self).__init__()
        self.pre = nn.Sequential(nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                           nn.BatchNorm2d(64), nn.ReLU(),
                           nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.layer1 = nn.Sequential(*self.make_layer(Residual, 64, 64, 3, first_block=True))
        self.layer2 = nn.Sequential(*self.make_layer(Residual, 64, 128, 4))
        self.layer3 = nn.Sequential(*self.make_layer(Residual, 128, 256, 6))
        self.layer4 = nn.Sequential(*self.make_layer(Residual, 256, 512, 3))
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, block, input_channels, num_channels, num_residuals, first_block=False):
        blk = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.append(block(input_channels, num_channels,
                                    use1x1conv=True, strides=2))
            else:
                blk.append(block(num_channels, num_channels))
        return blk

    def forward(self, x):
        x = self.pre(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)  #2x2x512

        x = F.avg_pool2d(x, 2)   #1x1x512
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    lr , epoch_num, batch_size = 0.0002, 20, 1000

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.465, 0.452, 0.387], std=[0.267, 0.258, 0.272])  # 正则化，四舍五入取三位
    ])
    train_dir ="../ImageData2/train"
    train_dataset = ImageFolder(root=train_dir, transform=train_transform)
    train_iter = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size, shuffle=True)
    print(f'train dataset length: {len(train_dataset)}')

    train_transform = transforms.Compose([
        # transforms.Resize((64,64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.464, 0.449, 0.386], std=[0.266, 0.259, 0.274])  # 正则化,四舍五入取三位
    ])
    valid_dir = "../ImageData2/val"
    valid_dataset = ImageFolder(root=valid_dir, transform=train_transform)
    valid_iter = torch.utils.data.DataLoader(valid_dataset,batch_size=batch_size, shuffle=True)
    print(f'val dataset length: {len(valid_dataset)}')

    net = ResNet34().to(device)
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    history = []
    best_acc = 0.0
    best_epoch= 0
    for epoch in range(epoch_num):
        epoch_start = time.time()
        print("epoch: {}/{}".format(epoch+1, epoch_num))
        net.train()
        train_loss = 0.0
        train_acc = 0.0
        valid_loss = 0.0
        valid_acc = 0.0
        for i,(inputs, labels) in enumerate(train_iter):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            l = loss(outputs, labels)
            l.backward()
            optimizer.step()

            train_loss += l.item() * inputs.size(0)
            ret, predictions = torch.max(outputs.data, 1)
            correct_counts = predictions.eq(labels.data.view_as(predictions))
            acc = torch.mean(correct_counts.type(torch.FloatTensor))
            train_acc+=acc.item() * inputs.size(0)
            # print('finished one batch')          #我的m1大约10s一个batch，哎，太难了
        with torch.no_grad():
            net.eval()
            for j, (inputs, labels) in enumerate(valid_iter):
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = net(inputs)
                l = loss(outputs, labels)
                valid_loss += l.item() * inputs.size(0)
                ret, predictions = torch.max(outputs.data, 1)
                correct_counts = predictions.eq(labels.data.view_as(predictions))
                acc = torch.mean(correct_counts.type(torch.FloatTensor))
                valid_acc += acc.item() * inputs.size(0)
        avg_train_loss = train_loss/len(train_dataset)
        avg_train_acc = train_acc/len(train_dataset)
        avg_valid_loss= valid_loss/len(valid_dataset)
        avg_valid_acc= valid_acc/len(valid_dataset)
        history.append([epoch+1,avg_train_loss,avg_valid_loss,avg_train_acc,avg_valid_acc])
        if best_acc < avg_valid_acc:
            best_acc = avg_valid_acc
            best_epoch = epoch+1
            # torch.save(net.state_dict(), "./best"+str(best_epoch)+".pt")
            torch.save(net.state_dict(), "./best.pt")
        epoch_end = time.time()
        print("training loss: {:.4f} , training acc: {:.4f}%, valid loss: {:.4f}, valid acc: {:.4f}% ,"
              "time: {:.4f}s".format(avg_train_loss, avg_train_acc*100, avg_valid_loss, avg_valid_acc*100, epoch_end-epoch_start))
        print("best acc: {:.4f}% at epoch: {:03d} ".format(best_acc*100, best_epoch))

    history = np.array(history)
    np.save('history.npy', history)
    plt.plot(history[:, 0], history[:, 1:3])
    plt.legend(['train loss', 'valid loss'])
    plt.xlabel('epoch num')
    plt.ylabel('loss')
    plt.ylim(0, 1.1)
    plt.savefig('loss_curve.png')
    # plt.show()

    plt.plot(history[:, 0], history[:, 3:5])
    plt.legend(['train acc', 'valid acc'])
    plt.xlabel('epoch num')
    plt.ylabel('acc')
    plt.ylim(0, 1.1)
    plt.savefig('acc_curve.png')
    # plt.show()




