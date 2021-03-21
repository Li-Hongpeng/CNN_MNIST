from torchvision import datasets, transforms


# 下载训练集与测试集,本项目已下载好,如需重新下载,将download值改为True
train_data_set = datasets.MNIST(root='./data/',
                               train=True,
                               transform=transforms.ToTensor(),
                               download=False)

test_data_set = datasets.MNIST(root='./data/',
                              train=False,
                              transform=transforms.ToTensor(),
                              download=False)


