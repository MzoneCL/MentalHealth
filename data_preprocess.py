import pandas as pd
import numpy as np
import os


raw_data_path = 'data_raw/data_faker.csv'


# 将数据中所有的空白字符(' ')转为''
def remove_blank():
    data_raw = pd.read_csv(raw_data_path)
    len_ = len(data_raw)
    cols = data_raw.columns
    for i in range(len_):
        for col in cols:
            if type(data_raw.iloc[i][col]) == str:
                data_raw.loc[i, col] = data_raw.loc[i, col].replace(' ', '')
    data_raw.to_csv(raw_data_path, encoding='utf_8_sig')


# 众数填充
def fill_with_mode():
    data_raw = pd.read_csv(raw_data_path)
    cols = data_raw.columns
    for col in cols:
        if data_raw[col].isnull().sum() > 0:
            print('填充：' + col)
            data_raw[col].fillna(data_raw[col].mode()[0], inplace=True)
    data_raw.to_csv('data_processed/fill_with_mode.csv', encoding='utf_8_sig')


# 将数据缩放到 min_ ~ max_ 范围内
def scale_data(df, min_, max_):
    pass


def normalized(df):
    return (df - df.min()) / (df.max() - df.min())


if __name__ == '__main__':
   # remove_blank()
    fill_with_mode()

