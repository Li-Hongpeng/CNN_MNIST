import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from model import LeNet
from data_loader import test_data_set, train_data_set
import argparse
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument("--LR", type=float, default=0.001)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=64)
arg = parser.parse_args()


if __name__ == '__main__':
    # 装载训练集
    train_loader = DataLoader(dataset=train_data_set, batch_size=arg.batch_size, shuffle=True)
    # 装载测试集
    test_loader = DataLoader(dataset=test_data_set, batch_size=arg.batch_size, shuffle=True)
    # 判断CUDA是否可用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = LeNet().to(device)
    # 使用交叉熵作为损失函数
    criterion = nn.CrossEntropyLoss()
    # 使用Adam自适应优化算法
    optimizer = optim.Adam(
        net.parameters(),
        lr=arg.LR,
    )

    for epoch in range(arg.epochs):
        sum_loss = 0.0
        for i, data in enumerate(train_loader):
            inputs, labels = data
            # 若CUDA可用 可将cpu改成CUDA
            inputs, labels = Variable(inputs).cpu(), Variable(labels).cpu()
            optimizer.zero_grad()  # 梯度归零
            outputs = net(inputs)  # 将数据传入网络进行前向运算
            loss = criterion(outputs, labels)  # 得到损失函数
            loss.backward()  # 反向传播
            optimizer.step()  # 优化更新
            sum_loss += loss.item()
            if i % 100 == 99:
                print('epoch:%d, step:%d, loss:%.03f' % (epoch + 1, i + 1, sum_loss / 100))
                sum_loss = 0.0
    print("train finished, model saved.")
    torch.save(net.state_dict(), './model_save/LeNet.pth')
    print("-----------------start testing-----------------")

    net.eval()  # 将模型变换为测试模式
    correct = 0
    total = 0
    for data_test in test_loader:
        images, labels = data_test
        images, labels = Variable(images).cpu(), Variable(labels).cpu()
        output_test = net(images)
        _, predicted = torch.max(output_test, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    print("Test finished. Test acc = {0}".format(correct.item() / len(test_data_set)))

