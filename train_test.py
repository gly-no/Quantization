import torch
import torchvision
import torch.nn as nn
import torch.utils.data as Data
from model import LeNet
from model import FCnet
from solve import W_Quan
from solve import W_grad

# model = LeNet()
model = FCnet()
# model.initialize()

#定义超参
Epoch = 10
batch_size = 64
lr = 0.1
M = 8  #量化比特数

#导入下载的数据集，这里只用训练集
train_data = torchvision.datasets.MNIST(root='./data/',train=True,transform=torchvision.transforms.ToTensor(),download=False)
train_loader = Data.DataLoader(train_data,batch_size = batch_size,shuffle=True,num_workers=0,drop_last=True)
loss_function = nn.CrossEntropyLoss()
# loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr=lr)
torch.set_grad_enabled(True)#在接下来的计算中每一次运算产生的节点都是可以求导的
model.train()
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(Epoch):
    running_loss = 0.0
    acc = 0.0
    for step, data in enumerate(train_loader):
        x,y = data
        optimizer.zero_grad()#实现梯度清零
        out = W_Quan(model=model, M=M)
        model = out[0]
        y_pred = model(x.to(device, torch.float))
        # pred =y_pred.argmax(dim=1)
        loss = loss_function(y_pred, y.to(device,torch.long))
        loss.backward()

        G = W_grad(model=model,grad=out[1:])
        model.F1.weight.grad = G[0].to(torch.float32)
        model.F2.weight.grad = G[1].to(torch.float32)
        model.OUT.weight.grad = G[2].to(torch.float32)

        # model.F1.weight.grad = torch.zeros(100,784)*model.F1.weight.grad
        # model.F2.weight.grad = torch.zeros(20,100)*model.F2.weight.grad
        # model.OUT.weight.grad = torch.zeros(10,20)*model.OUT.weight.grad

        running_loss += float(loss.data.cpu())
        pred =y_pred.argmax(dim=1)
        acc += (pred.data.cpu() == y.data).sum()
        optimizer.step()
        if step % 100 == 99:
            loss_avg = running_loss / (step + 1)
            acc_avg = float(acc / ((step + 1) * batch_size))
            print('Epoch', epoch+1, ',step', step+1, '| Loss_avg: %.4f' % loss_avg, '|Acc_avg:%.4f' % acc_avg)


torch.save(model, './FCnet_Quan_test.pkl')
