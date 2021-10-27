import numpy as np
import pandas as pd
# 导入评价子模块
from sklearn import metrics
from RemarkModel import evaluate_model

def WeigtingRegress(modelList,X_test,y_test,allX):
    '''计算模型权重，并集成
        accList: 各模型的准确得分【测试集准确率等】
        perdYList: 各模型在*全体*数据集下获得的预测值列表
    '''
    accList,predYList = getList(modelList,X_test,y_test,allX)
    accList = np.array(accList)

    # 归一，得到各模型权重
    weights = (accList/accList.sum()).reshape(len(modelList),1)
    
    return np.sum((weights*predYList),axis=0)

def getList(modelList,X_test,y_test,allX):
    accList = []
    predYList = []
    for model in modelList:
        y_pred = model.predict(X_test)
        # 回归值转为01分类值（四舍五入）
        y_pred = np.array([round(i) for i in y_pred])
        # 用测试集的正确率作为权重基准
        accList.append(metrics.accuracy_score(y_test, y_pred))
        predYList.append(model.predict(allX))
    return accList,predYList

def trainTogether(modelList,X_train,y_train):
    for i,model in enumerate(modelList):
        model.fit(X_train,y_train)
        modelList[i] = model    # 使用该方法，才能更新列表中的变化
    return modelList

if __name__ == "__main__":
    
    # 先创建基本模型列表
    import xgboost as xgb
    from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
    from sklearn.svm import SVC
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import SGDRegressor
    # from network import BN 
        # 我这儿的环境两个包不兼容，懒得重新整了，所以就不能放在一起，可能得在你那儿跑一下
    '''modelList = [RandomForestClassifier(200, random_state=0),
                  xgb.XGBClassifier(), GradientBoostingClassifier(),
                  SVC(kernel='sigmoid'),SGDRegressor()]'''
    modelList = [SVC(kernel='sigmoid',probability=True),
                  SVC(kernel='sigmoid',probability=True,coef0=0.5),
                  SVC(C=0.5,kernel='sigmoid',probability=True),     # 虽然不用参数下会略微过拟合，但是0.1的C下又欠拟合...
                  # 前三个模型都很拉，不要sigmoid算了
                  SVC(kernel='linear',probability=True),
                  SVC(kernel='linear',probability=True,gamma=0.6), 
                  SVC(kernel='linear',probability=True,gamma=0.8),
                  # 测试后，tol这个参数基本没什么用，gamma 这个参数的区别也不大
                  SVC(probability=True),
                  SVC(probability=True,gamma=0.6),
                  SVC(probability=True,gamma=0.8)]

    dataset = pd.read_csv('./data/attr_std.csv')
    X = dataset.iloc[:, 2:-1].values
    y = dataset.iloc[:, -1].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    
    modelList = trainTogether(modelList,X_train,y_train)
    '''rlt = WeigtingRegress(modelList,X_test,y_test,X)
    print(rlt)'''
    evaluate_model(modelList,X_test,y_test).to_csv('test_SVM.csv')
    evaluate_model(modelList,X_train,y_train).to_csv('train_SVM.csv')
    