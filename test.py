import torch
import torchvision
import torch.nn as nn
import torch.utils.data as Data
from model import LeNet
from model import FCnet
from solve import W_Quan
from solve import W_grad

test_data = torchvision.datasets.MNIST(root='./data/', train=False,transform=torchvision.transforms.ToTensor(),download=False)
test_loader = Data.DataLoader(test_data,batch_size=1,shuffle=False)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
net = torch.load('./FCnet_Quan_8bits_lr0.01.pkl', map_location=torch.device(device))

# M = 2  #量化比特数
net.to(device)
# out = W_Quan(model=net, M=M)
# net = out[0]
#关闭梯度、屏蔽dropout层、冻结BN层的参数，防止测试阶段BN发生参数更新
torch.set_grad_enabled(False)
net.eval()

#开始测试
length = test_data.data.size(0)
acc = 0.0

for i, data in enumerate(test_loader):
    x, y = data
    y_pred = net(x.to(device, torch.float))
    pred = y_pred.argmax(dim=1)
    acc += (pred.data.cpu() == y.data).sum()
    # print(i)
    # print('Predict:', int(pred.data.cpu()), '|Ground Truth:', int(y.data))
acc = (acc / length) * 100
print('Accuracy: %.2f' %acc, '%')

