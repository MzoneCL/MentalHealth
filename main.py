from xgboost import XGBClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from sklearn.linear_model import LogisticRegression

from data_preprocess import normalized


data_mode_path = 'data_processed/fill_with_mode.csv'

def classification_report(Y_true, Y_pred):
    print(precision_score(Y_true, Y_pred))
    print(recall_score(Y_true, Y_pred))
    print(f1_score(Y_true, Y_pred))
    print(confusion_matrix(Y_true, Y_pred))

def get_data():
    data = pd.read_csv(data_mode_path)
    data_X = data.drop(['学号/工号', '姓名', 'y_label'], axis=1)
    data_Y = data['y_label']

    data_X['性别'] = data_X['性别'].map({'男': 1, '女': 0})

   # print(data_X.info())

    data_X = normalized(data_X) # 归一化

    X = np.array(data_X)
    Y = np.array(data_Y)

    return X, Y


def use_xgb():
    X, Y = get_data()

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

    xgb = XGBClassifier()
    xgb.fit(X_train, Y_train)

    Y_pred = xgb.predict(X_test)

    classification_report(Y_test, Y_pred)


def use_lr():
    X, Y = get_data()

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

    lr = LogisticRegression()
    lr.fit(X_train, Y_train)

    Y_pred = lr.predict(X_test)

    classification_report(Y_test, Y_pred)

if __name__ == '__main__':
    print('begin')
    use_lr()
