import torch
import numpy as np
import torch.nn as nn #包含网络的基类，比如卷积层、池化层、激活函数、全连接层等，损失函数的调用


#定义一个类，这个类继承nn.module
class FCnet(nn.Module):
    def __init__(self):
        super(FCnet,self).__init__()
        self.F1 = nn.Linear(784,100)
        self.R1 = nn.ReLU()
        self.D1 = nn.Dropout(0.3)
        self.F2 = nn.Linear(100,20)
        self.R2 = nn.ReLU()
        self.D2 = nn.Dropout(0.2)
        self.OUT = nn.Linear(20,10)
        self.S = nn.Softmax(1)
    def forward(self,x):
        x = x.view(x.size(0),-1)
        x = self.F1(x)
        x = self.R1(x)
        x = self.D1(x)
        x = self.F2(x)
        x = self.R2(x)
        x = self.D2(x)
        x = self.OUT(x)
        x = self.S(x)
        return x
    def initialize(self):
        for m in self.modules():
            if isinstance(m,nn.Linear):
                nn.init.normal_(m.weight.data)

class LeNet(nn.Module):
    #对模型进行初始化和前向传播
    def __init__(self):
        super(LeNet,self).__init__()#super是什么?
        self.C1 = nn.Conv2d(1,6,5,1,2)
        self.R1 = nn.ReLU()
        self.S2 = nn.MaxPool2d(2)
        self.C3 = nn.Conv2d(6,16,5,1,0)
        self.R2 = nn.ReLU()
        self.S4 = nn.MaxPool2d(2)
        self.C5 = nn.Conv2d(16,120,5,1,0)
        self.R3 = nn.ReLU()
        self.F6 = nn.Linear(120,84)
        self.R4 = nn.ReLU()
        self.OUT = nn.Linear(84,10)
        # nn.MaxPool2d(2*2,2,0)
        # nn.ReLU()
    def forward(self,x):
        x = self.C1(x)
        x = self.R1(x)
        x = self.S2(x)
        x = self.C3(x)
        x = self.R2(x)
        x = self.S4(x)
        x = self.C5(x)
        x = self.R3(x)
        x = x.view(x.size(0), -1)#全连接层只能输入二维数据
        x = self.F6(x)
        x = self.R4(x)
        x = self.OUT(x)
        return x

        
# A test!
if __name__ == "__main__":
    model = FCnet()
    a = torch.randn(1, 1, 28, 28)
    b = model(a)
    print(b)
