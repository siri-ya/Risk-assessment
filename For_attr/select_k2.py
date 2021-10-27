from sklearn.feature_selection import SelectKBest,chi2,mutual_info_regression,f_regression
import numpy as np
import pandas as pd

data = pd.read_csv('attr_std.csv')
label = np.array(data.iloc[:, -1])
data = np.array(data.drop(data.columns[[0, -1]], axis=1))


#选择K个最好的特征，返回选择特征后的数据
def SelectAttr(func):
    model = SelectKBest(func, k=10)
    X_new = model.fit_transform(data, label)
    scores = model.scores_
    indices = np.argsort(scores)
    print(indices)

if __name__ == "__main__":
    functions = [chi2,mutual_info_regression,f_regression]
    for func in functions:
        SelectAttr(func)