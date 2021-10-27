import numpy as np
import pandas as pd
from scipy.stats import f

# 读取标准化数据
ori_df = pd.read_csv('./data/attr_std.csv')
df = [ori_df[ori_df.result == 0].copy(), ori_df[ori_df.result == 1].copy()]  # df[0]负样本，df[1]正样本
p_num, n_num = df[1]['id'].count(), df[0]['id'].count()
attrs = list(ori_df.columns)
attrs.remove('id')
attrs.remove('result')
print("一共有"+str(len(attrs))+"个指标")

# 计算每个特征F检验值
attr_F = {}
for attr in attrs:
    avg_d, sum_D = [0, 0, 0], [0, 0]
    for i in range(2):
        meann = df[i][attr].mean()
        df[i][attr+'_d'] = abs(df[i][attr] - meann)
        avg_d[i] = df[i][attr+'_d'].mean()
        df[i][attr+'_d'] = pow(df[i][attr+'_d']-avg_d[i], 2)
        sum_D[i] = df[i][attr+'_d'].sum()
    avg_d[2] = (avg_d[0]+avg_d[1])/2
    nume = (p_num+n_num-2)*(n_num*pow(avg_d[0]-avg_d[2], 2)+p_num*pow(avg_d[1]-avg_d[2], 2))  # F检验值分子
    deno = sum_D[0]+sum_D[1]  # F检验值分母
    attr_F[attr] = nume/deno  # 计算F检验值

# 淘汰F值较小者
attr_F = pd.Series(attr_F)
print(attr_F)
threshold = f.isf(0.01, 1, 121)  # 筛选值
attr_out = attr_F[attr_F <= threshold]
print("第一次筛选的被淘汰指标："+str(len(attr_out))+"个")
print(attr_out)
attr_F = attr_F[attr_F > threshold]
attrs_ok = list(attr_F.index)

# 去除冗余前的准备
attr_left = pd.DataFrame(pd.Series(1, index=attrs_ok), columns=['selected'])
attr_left['F_score'] = attr_F
for attr in attrs_ok:
    mean = ori_df[attr].mean()
    attr_left.loc[attr, 'mean'] = mean
    ori_df[attr+'_d'] = ori_df[attr] - mean
    ori_df[attr+'_D'] = pow(ori_df[attr] - mean, 2)

# 计算相关系数，删除冗余
group_checked = []
for attr1 in attrs_ok:
    for attr2 in attrs_ok:
        if attr1 != attr2 and attr_left.loc[attr1, 'selected'] == 1 and attr_left.loc[attr1, 'selected'] == 1:
            if (attr1, attr2) not in group_checked:
                group_checked.append((attr2, attr1))
                ori_df[attr1 + attr2] = ori_df[attr1 + '_d'] * ori_df[attr2 + '_d']
                nume = ori_df[attr1 + attr2].sum()
                deno = pow(ori_df[attr1 + '_D'].sum() * ori_df[attr2 + '_D'].sum(), 0.5)
                r = nume / deno
                if r > 0.8:
                    print("\n发现高度相关变量：" + attr1 + ", " + attr2 + ", r= " + str(r))
                    if attr_left.loc[attr1, 'F_score'] >= attr_left.loc[attr2, 'F_score']:
                        attr_left.loc[attr2, 'selected'] = 0
                    else:
                        attr_left.loc[attr1, 'selected'] = 0
out = attr_left[attr_left['selected'] == 0].copy()
print("\n\n第二次被淘汰的指标如下："+str(out['selected'].count())+"个")
print(out)
left = attr_left[attr_left['selected'] == 1].copy()
print("\n\n剩余指标如下："+str(left['selected'].count())+"个")
print(left)
left = ori_df[left.index].copy()
left['result'] = ori_df['result']
left.index = ori_df['id']
left.to_csv('./data/attr_selected.csv')

# 计算相关系数














