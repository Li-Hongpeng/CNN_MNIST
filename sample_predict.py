import torch
from model import LeNet
from PIL import Image
import numpy as np
import argparse

def img2MNIST(filename):
    img = Image.open(filename).convert('L')
    img = img.resize((28,28),Image.ANTIALIAS)
    arr = []
    for i in range(28):
        for j in range(28):
            pixel = float(img.getpixel((j, i))) / 255.0
            arr.append(pixel)
    arr1 = np.array(arr).reshape((1,1,28,28))
    result = torch.as_tensor(arr1, dtype=torch.float32)
    return result


parser = argparse.ArgumentParser()
parser.add_argument("--filename", type=str, default='./test_sample.bmp')
parser.add_argument("--model", type=str, default='./model_save/LeNet.pth')
arg = parser.parse_args()


if __name__ == "__main__":
    print("Tested picture " + arg.filename)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_load = LeNet()
    model_load.load_state_dict(torch.load(arg.model))
    net = model_load.to(device)
    net.eval()
    image = img2MNIST(arg.filename)
    output_test = net(image)
    _, predicted = torch.max(output_test, 1)
    print("The hand writing number is: " + str(predicted.item()))
