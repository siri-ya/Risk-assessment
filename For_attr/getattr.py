from datetime import datetime
import pandas as pd
from numpy import inf, log


def OverallScale(dfsrc, name):
    """总体数值规模计算"""
    # 作废发票的计数、金额和、税额和
    rlt1 = dfsrc[dfsrc['发票状态'] == '作废发票'].groupby('企业代号', sort=False).agg(
        {'发票号码': 'count', '金额': 'sum', '税额': 'sum'})
    rlt1.columns = ['cancel_cnt' + name, 'cancel_mny' + name, 'cancel_tax' + name]

    # 有效发票的计数、金额和、税额和
    rlt2 = dfsrc[dfsrc['发票状态'] == '有效发票'].groupby('企业代号', sort=False).agg(
        {'发票号码': 'count', '金额': 'sum', '税额': 'sum'})
    rlt2.columns = ['valid_cnt' + name, 'valid_mny' + name, 'valid_tax' + name]

    # 将二表依照索引进行连接
    rlt = pd.concat([rlt2, rlt1], axis=1)

    # 顺便计算部分稳定性
    rlt['cancel_cnt_rate' + name] = rlt['cancel_cnt' + name] / (rlt['cancel_cnt' + name] + rlt['valid_cnt' + name])
    rlt['cancel_mny_rate' + name] = rlt['cancel_mny' + name] / (rlt['cancel_mny' + name] + rlt['valid_mny' + name])
    rlt['negative_rate' + name] = dfsrc.groupby('企业代号', sort=False)['金额'].agg(lambda x: sum(x < 0)) / rlt[
        'valid_cnt' + name]

    # 有的企业没有作废发票，因此需将none变为0
    rlt.fillna(0, inplace=True)
    return rlt


def predict(df_slice):
    """估计近期总利润/负债"""
    df_slice = df_slice.reset_index()
    gap = [datetime(2019, 8, 1), datetime(2019, 9, 1), datetime(2019, 10, 1), datetime(2019, 11, 1),
           datetime(2019, 12, 1), datetime(2020, 1, 1), datetime(2020, 2, 1)]
    weight = [0.05, 0.1, 0.15, 0.15, 0.25, 0.3]
    pred = 0
    for i in range(6):
        t = df_slice[df_slice['开票日期'] >= gap[i]]
        t = t[t['开票日期'] < gap[i + 1]]
        pred += t.iloc[:, -1].sum() * weight[i]
    return pred


def count_growth(df_slice):
    """计算近期的半年平均增长率"""
    df_slice = df_slice.reset_index()
    gap = [datetime(2018, 1, 1), datetime(2018, 7, 1), datetime(2019, 7, 1), datetime(2020, 1, 1)]
    x = []
    for i in range(2):
        t = df_slice[df_slice['开票日期'] >= gap[2 * i]]
        t = t[t['开票日期'] < gap[2 * i + 1]]
        x.append(t.iloc[:, -1].sum())  # 求半年的总盈利
    return pow(x[1] / x[0], 1 / 3) - 1 if x[0] != 0 else 2  # 返回两年的半年平均增长率


def count_acceleration(df_slice):
    """计算近期的半年增长加速度"""
    df_slice = df_slice.reset_index()
    gap = [datetime(2018, 1, 1), datetime(2018, 7, 1), datetime(2019, 1, 1), datetime(2019, 7, 1), datetime(2020, 1, 1)]
    x, v = [], []
    for i in range(4):
        t = df_slice[df_slice['开票日期'] >= gap[i]]
        t = t[t['开票日期'] < gap[i + 1]]
        x.append(t.iloc[:, -1].sum())  # 求每半年的总盈利
        if i > 0:
            v.append(x[i] / x[i - 1] - 1 if x[i - 1] != 0 else 2)  # 求每半年的增长量
    return (v[2] - v[0]) / 2  # 返回半年的平均增长加速度


def Vary(df1, df2):
    """时间维度计算"""
    df1['开票日期'] = df1['开票日期'].apply(lambda x: x.strftime('%Y-%m')).astype('datetime64')
    df2['开票日期'] = df2['开票日期'].apply(lambda x: x.strftime('%Y-%m')).astype('datetime64')

    # 求每个月的总利润
    month_i = df1[df1['发票状态'] == '有效发票'].groupby(['企业代号', '开票日期'], sort=False).agg({'金额': 'sum'})
    month_o = df2[df2['发票状态'] == '有效发票'].groupby(['企业代号', '开票日期'], sort=False).agg({'金额': 'sum'})
    month_i['month_profit'] = month_o['金额'] - month_i['金额']

    # 求近期收益估计值、近两年总体增长率、半年平均增长加速度
    tend = month_i.groupby(['企业代号'], sort=False).agg({'month_profit': 'sum'})
    tend['profit_estimation'] = month_i.groupby(['企业代号'], sort=False).apply(predict)
    tend['p_growth_rate'] = month_i.groupby(['企业代号'], sort=False).apply(count_growth)
    tend['p_acceleration'] = month_i.groupby(['企业代号'], sort=False).apply(count_acceleration)
    tend.drop(['month_profit'], axis=1, inplace=True)

    # 求每个月的总负债
    month_i = df1[df1['发票状态'] == '有效发票'].groupby(['企业代号', '开票日期'], sort=False).agg({'税额': 'sum'})
    month_o = df2[df2['发票状态'] == '有效发票'].groupby(['企业代号', '开票日期'], sort=False).agg({'税额': 'sum'})
    month_i['month_tax'] = month_o['税额'] - month_i['税额']

    # 求近期负债估计值、及其近两年总体增长率、半年平均增长加速度
    tend['tax_estimation'] = month_i.groupby(['企业代号'], sort=False).apply(predict)
    tend['t_growth_rate'] = month_i.groupby(['企业代号'], sort=False).apply(count_growth)
    tend['t_acceleration'] = month_i.groupby(['企业代号'], sort=False).apply(count_acceleration)

    # 求近期极润率，负债率，总票据流量(进项+销项,之所以加起来是因为不管进货还是出货，都能体现公司一定的经济活力)
    t = df1[df1['发票状态'] == '有效发票']
    t = t[t['开票日期'] >= datetime(2019, 2, 1)]
    t = t[t['开票日期'] < datetime(2020, 2, 1)]
    tend['temp1'] = t.groupby(['企业代号'], sort=False).agg({'金额': 'sum'})
    tend['temp2'] = t.groupby(['企业代号'], sort=False).agg({'税额': 'sum'})
    tend['bill_cnt_in'] = t.groupby(['企业代号'], sort=False).agg({'发票状态': 'count'})

    t = df2[df2['发票状态'] == '有效发票']
    t = t[t['开票日期'] >= datetime(2019, 2, 1)]
    t = t[t['开票日期'] < datetime(2020, 2, 1)]
    tend['bill_cnt_out'] = t.groupby(['企业代号'], sort=False).agg({'发票状态': 'count'})
    tend['bill_cnt_year'] = tend['bill_cnt_in'] + tend['bill_cnt_out']
    tend['temp3'] = t.groupby(['企业代号'], sort=False).agg({'金额': 'sum'})
    tend['p_margin_year'] = tend['temp3'] - tend['temp1']
    tend['t_margin_year'] = t.groupby(['企业代号'], sort=False).agg({'税额': 'sum'})
    tend['t_margin_year'] = (tend['t_margin_year'] - tend['temp2'])/tend['p_margin_year']
    tend['p_margin_year'] = tend['p_margin_year']/tend['temp3']


    tend.fillna(0, inplace=True)
    return tend.drop(['temp1', 'temp2', 'temp3'], axis=1)


def kkurt(df_slice):
    """求加权峰度系数"""
    rlt = df_slice.groupby(['开票日期'], sort=False).agg({'价税合计': 'count'}).kurt()
    rlt.fillna(-1.2, inplace=True)
    return rlt


def Stability(dfsrc, another, order):
    """稳定性计算"""
    suffix = '_in' if another == "销方" else '_out'
    # 以月为单位计算稳定性
    dfsrc['开票日期'] = dfsrc['开票日期'].apply(lambda x: x.strftime('%Y-%m')).astype('datetime64')
    if order == '_year':
        dfsrc = dfsrc[dfsrc['开票日期'] >= datetime(2019, 2, 1)]
        dfsrc = dfsrc[dfsrc['开票日期'] < datetime(2020, 2, 1)]
        suffix += order

    # 合作伙伴稳定性: 公司与各单位的合作次数峰度的加权均值
    focus = dfsrc[dfsrc['发票状态'] == '有效发票']
    partner_kurt = focus.groupby(['企业代号', another + '单位代号'], sort=False).apply(kkurt)
    partner_cnt = focus.groupby(['企业代号', another + '单位代号'], sort=False).agg({'价税合计': 'count'})
    partner_kurt['kurt'] = partner_cnt['价税合计'] * partner_kurt['价税合计']
    stability = partner_kurt.groupby(['企业代号'], sort=False).agg({'kurt': 'sum'})
    all_cnt = focus.groupby(['企业代号'], sort=False).agg({'价税合计': 'count'})
    stability['partner_kurt' + suffix] = stability['kurt'] / all_cnt['价税合计']
    stability.fillna(-1.2, inplace=True)
    stability.replace("", -1.2, inplace=True)

    # 票据总额稳定性：用标准方差表示波动大小
    stability['mny_wave' + suffix] = 0
    stability['mny_wave' + suffix] = focus[focus['价税合计'] > 0].groupby('企业代号', sort=False)['价税合计'].std()
    stability.fillna(0, inplace=True)
    stability.replace("", 0, inplace=True)

    # 票据时间稳定性：
    period = focus.groupby(['企业代号', '开票日期'], sort=False).agg({'价税合计': 'sum'})
    stability['time_wave' + suffix] = period.groupby(['企业代号'], sort=False)['价税合计'].apply(lambda x: x.kurt())
    stability.fillna(-1.2, inplace=True)
    stability.replace("", -1.2, inplace=True)
    return stability.drop(['kurt'], axis=1)


def Credit():
    """信用评级转化为分数"""
    # 需要设定索引为企业代号列，否则之后链接时出错
    df = pd.read_excel('./data/file1.xlsx', sheet_name='企业信息').set_index('企业代号')['信誉评级']
    return df.replace(to_replace=['A', 'B', 'C', 'D'], value=[30, 20, 10, 0])

def extractinfo(x: str):
    if '福利院' in x or '消防' in x:
        x = '第四产业'

    if '科技' in x or '研究' in x or '技术' in x or '勘测' in x \
            or '设计' in x or '传媒' in x or '娱乐' in x or '文化' in x \
            or '管理' in x or '发展' in x or '土地' in x or '广告' in x:

        x = '脑力型第三产业'

    elif '贸' in x or '劳务' in x or '经营' in x or '店' in x or '销售' in x \
            or '咨询' in x or '实业' in x or '图书' in x or '物流' in x \
            or '租赁' in x or '药' in x or '美容' in x or '园艺' in x:

        x = '体力型第三产业'

    elif '花' in x or '农' in x or '林' in x or '木' in x:
        x = '第一产业'

    else:
        x = '第二产业'

    return x

def get_WOE():
    df = pd.read_excel('./data/file1.xlsx', sheet_name='企业信息', index_col=0)[['企业名称', '是否违约']]
    df['是否违约'].replace(['是', '否'], [1, 0], inplace=True)
    defalutsum = sum(df['是否违约'])
    benignsum = df['是否违约'].count() - defalutsum
    # 利用企业名对其进行划分
    df['企业种类'] = df['企业名称'].str.strip().apply(extractinfo)
    # 进行WOE编码
    WOEencoding = df.groupby('企业种类', sort=False)['是否违约'].agg(
        lambda x: log((sum(x) / defalutsum) / (sum(x == 0) / benignsum)))
    # 连接
    df = pd.merge(df, WOEencoding.rename('type_WOE'), how='left', on='企业种类')
    df['type_WOE'].to_csv('WOEResult.csv')
    df['type_WOE'].replace(float('-inf'), -2, inplace=True)
    return pd.Series(df['type_WOE'])


def perprocess1(datapath, problem):
    """特征计算"""
    df1 = pd.read_excel(datapath, sheet_name='进项发票信息')
    df2 = pd.read_excel(datapath, sheet_name='销项发票信息')
    print("ok")
    # 数值规模
    attrDF1 = OverallScale(df1, '_in')
    attrDF2 = OverallScale(df2, '_out')
    attr = pd.concat([attrDF1, attrDF2], axis=1)
    attr['profit'] = attr['valid_mny_out'] - attr['valid_mny_in']
    attr['debt'] = attr['valid_tax_out'] - attr['valid_tax_in']
    attr['profit_margin'] = attr['profit'] / attr['valid_mny_out']
    attr['debt_margin'] = attr['debt'] / attr['profit']
    print("ok")
    # 时间序列
    tendency = Vary(df1, df2)
    attr = pd.concat([attr, tendency], axis=1)
    print("ok")
    # 稳定性
    stab1 = Stability(df1, '销方', '_all')
    stab2 = Stability(df2, '购方', '_all')
    stab = pd.concat([stab1, stab2], axis=1)
    attr = pd.concat([attr, stab], axis=1)
    print("ok")
    stab1 = Stability(df1.copy(), '销方', '_year')
    stab2 = Stability(df2.copy(), '购方', '_year')
    stab = pd.concat([stab1, stab2], axis=1)
    attr = pd.concat([attr, stab], axis=1)
    attr.fillna(0, inplace=True)
    print("ok")
    # 信用评级
    if problem != 2:
        attr['credit_score'] = Credit()
    # 输出
    print("ok")
    # 这里还是不对，我没明白
    attr['type_WOE'] = get_WOE()
    attr.reset_index()
    attr.rename({'企业代号': 'id'}, axis=1)
    attr.to_csv('./data/attr.csv')


if __name__ == "__main__":
    perprocess1('./data/file1.xlsx', 1) # 1代表问题1