import pandas as pd
import numpy as np

def getdata():
    data = pd.read_csv('data/attr_std.csv',index_col=1)
    label = np.array(data.iloc[:,-1])
    return data,label

def AdaBoost(data,label,modelist):
    # 初始化分布D
    D = np.ones((123,1))/123
    aggregate_weight = np.zeros((123,1))
    weights = []
    for model in modelist:
        # 如果是神经网络的话，train函数中需要写好从data至dataloader的转换
        y = pd.Series(model.train(data,label,D))
        wrongRate = 1 - (label==y).sum()/len(label)
        alpha = np.log((1-wrongRate)/max(wrongRate,1e-5))/2
        weights.append(alpha)
        # 更新D
        expon = np.where(y==label,np.ones(123)*(-alpha),np.ones(123)*alpha).reshape(123,1)
        D = np.exp(expon)*D
        D = D/D.sum()
        # 给所有弱学习器的预测结果加上权重
        aggregate_weight += alpha*y
        sign_predict = np.sign(aggregate_weight)
        aggregate_error = (np.multiply(sign_predict!=label,np.ones((123,1)))).sum()/123
        if aggregate_error == 0:
            break
    return modelist,sign_predict,weights
    
if __name__ == "__main__":
    testD = np.random.random(123).reshape(123,1)
    testlabel = np.random.random(123).reshape(123,1)
    print(testD)
    print(testlabel)
    print(testD*testlabel)