import numpy as np
import pandas as pd

df = pd.read_csv('./data/attr.csv')
# woe = pd.read_csv('./data/WOEResult.csv')
# df['type_WOE'] = woe['type_WOE']
if_break = pd.read_excel('./data/file1.xlsx', sheet_name='企业信息')
df['result'] = if_break['是否违约']
df['result'] = df['result'].map({"否": 0, "是": 1})
std_df = df.copy()

"""新加的属性必须在这里过一下正负向，将正向指标放入列表！！！！！！！！！！！！！！！！！！！"""
positive = ['valid_cnt_in', 'valid_mny_in', 'valid_tax_in', 'valid_cnt_out', 'valid_mny_out', 'profit',
            'profit_estimation', 'profit_margin', 'p_growth_rate', 'p_acceleration', 'bill_cnt_in', 'bill_cnt_out',
            'bill_cnt_year', 'p_margin_year', 'credit_score', 'type_WOE', 'result']

column_list = list(std_df.columns)
for column in column_list[1:-1]:
    mi, ma = df[column].min(), df[column].max()
    dis = ma-mi
    if column in positive:
        std_df[column] = (df[column] - mi) / dis
    else:
        std_df[column] = (ma - df[column]) / dis
std_df.to_csv('./data/attr_std.csv', index=False)
