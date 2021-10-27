from numpy import inf, log
import pandas as pd


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


def processs():
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
    df['type_WOE'].to_csv('./data/WOEResult.csv')
    df['type_WOE'].replace(float('-inf'), -1, inplace=True)
    return df['type_WOE']


if __name__ == "__main__":
    processs()