import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
import numpy as np
from tqdm import tqdm

class mydata(Dataset):
    def __init__(self,choice,data=None,label=None) -> None:
        super(mydata,self).__init__()
        if choice:
            self.data = torch.from_numpy(np.load('./data/X_'+choice+'.npy')).float()
            # 使用交叉熵损失函数时需要类型为float
            self.label = torch.from_numpy(np.load('./data/y_'+choice+'.npy')).float()
        else:
            self.data = torch.from_numpy(data).float()
            self.label = torch.from_numpy(label).float()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index],self.label[index]

# 前馈神经网络
class BN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(BN, self).__init__()
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.l2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out = self.relu(self.l1(x))
        out = self.sigmoid(self.l2(out))
        return out

    # 集成接口1
    def fit(self,data,label):
        loader = DataLoader(
            mydata(data,label),
            batch_size = 3,
            shuffle = True,  # 是否打乱数据
            num_workers = 0  # 多线程来读数据，在Win下需要设置为0
        )
        train(self,8,torch.optim.Adam(self.parameters()),loader)

    # 集成接口2
    def predict(self,data):
        return self(data)

loss = torch.nn.BCELoss()  # 二分类交叉熵损失函数

def evaluate(data_iter, net):
    right_sum, n, loss_sum = 0.0, 0, 0.0
    for x,y in data_iter:
        y_ = net(x)
        l = loss(y_, y.view(-1,1))
        n += 1
        loss_sum += l.item()
    return loss_sum / n

def train(net, num_epochs, optimizer, dataloader):

    for epoch in range(num_epochs):
        right_num,train_l, n = 0,0.0, 0
        for X,y in tqdm(dataloader['train']):
            y_hat = net(X)
            n += y.shape[0]
            y = y.view(-1,1)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l += l.item()
            right_num += y.eq(torch.where(y_hat<0.5,0,1)).sum()
        print('epoch %d, train loss %.4f, train accuracy:%.4f' % (epoch + 1, train_l / n, right_num/n))
        test_l = evaluate(dataloader['test'], net)
        print('test loss', test_l)


if __name__ == "__main__":

    dataloader = {
        x : DataLoader(
        dataset = mydata(x),  # torch TensorDataset format
        batch_size = 3,
        shuffle = True,  # 是否打乱数据
        num_workers = 0  # 多线程来读数据，在Win下需要设置为0
    )
    for x in ['train','test']
    }

    net = BN(31,8)
    optimizer = torch.optim.Adam(net.parameters())
    train(net,8,optimizer,dataloader)
    print("保存模型中....")
    torch.save(net.state_dict(), "./model_save/model_default.pth")
    print('完成!')
