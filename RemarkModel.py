from sklearn import metrics
import numpy as np
import pandas as pd

def get_score(Model, X_test, y_test):
    """评估模型的预测性能"""
    # 代入测试集，得到预测值（[0~1]连续值）
    try:
        y_pred = Model.predict_proba(X_test)[:, 1]
    except:
        y_pred = Model.predict(X_test)
    # 回归指标值的评价：
    # MAE
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    # MSE
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    # RMSE
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    # KS
    ks_df = pd.crosstab(y_pred, y_test)
    ks_df = ks_df.cumsum(axis=0) / ks_df.sum()
    ks_df['gap'] = abs(ks_df[0] - ks_df[1])
    ks = ks_df['gap'].max()
    print("KS", ks)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)

    # 回归值转为01分类值（四舍五入）
    y_pred = np.array([round(i) for i in y_pred])
    # 分类指标值的评价：
    # 总正确率
    print('Accuracy', metrics.accuracy_score(y_test, y_pred))
    # 精确率
    print('Precision', metrics.precision_score(y_test, y_pred))
    # 召回率
    print('Recall', metrics.recall_score(y_test, y_pred))
    # 特效度（关注对负例的识别能力）
    tn, fp = metrics.confusion_matrix(y_test, y_pred)[0]
    print('specificity', tn / (tn + fp))
    # F1分数
    print('f1', metrics.f1_score(y_test, y_pred))
    # AUC
    print('Roc_Auc', metrics.auc(fpr, tpr))

def evaluate_model(modelList, X_test, y_test):
    """输入模型组modellist，测试集X_test，测试结果y_test, 返回评分dataframe"""
    scores = pd.DataFrame(np.random.rand(10,1), columns=['model1'],
                          index=['MAE','MSE','RMSE','KS','Acc','Pre','Rec','Spe','F1','Auc'])
    for i, model in enumerate(modelList):
        try:
            y_pred = model.predict_proba(X_test)[:, 1]
        except:
            y_pred = model.predict(X_test)
        # 回归指标值的评价：
        scores.loc['MAE', 'model'+str(i+1)] = metrics.mean_absolute_error(y_test, y_pred)
        scores.loc['MSE', 'model'+str(i+1)] = metrics.mean_squared_error(y_test, y_pred)
        scores.loc['RMSE', 'model'+str(i+1)] = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
        ks_df = pd.crosstab(y_pred, y_test)
        ks_df = ks_df.cumsum(axis=0) / ks_df.sum()
        scores.loc['KS', 'model'+str(i+1)] = abs(ks_df[0] - ks_df[1]).max()
        fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
        # 回归值转为01分类值（四舍五入）
        y_pred = np.array([round(i) for i in y_pred])
        # 分类指标值的评价：
        scores.loc['Acc', 'model'+str(i+1)] = metrics.accuracy_score(y_test, y_pred)
        scores.loc['Pre', 'model'+str(i+1)] = metrics.precision_score(y_test, y_pred)
        scores.loc['Rec', 'model'+str(i+1)] = metrics.recall_score(y_test, y_pred)
        tn, fp = metrics.confusion_matrix(y_test, y_pred)[0]
        scores.loc['Spe', 'model'+str(i+1)] = tn / (tn + fp)
        scores.loc['F1', 'model'+str(i+1)] = metrics.f1_score(y_test, y_pred)
        scores.loc['Auc', 'model'+str(i+1)] = metrics.auc(fpr, tpr)
    return scores
