from xgboost import XGBClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from data_preprocess import normalized


data_mode_path = 'data_processed/fill_with_mode.csv'

def classification_report(method: str, Y_true, Y_pred):
    print(method)
    print('Precision: ' + str(precision_score(Y_true, Y_pred)))
    print('Recall: ' + str(recall_score(Y_true, Y_pred)))
    print('F1: ' + str(f1_score(Y_true, Y_pred)))
    print(confusion_matrix(Y_true, Y_pred))
    print()

def get_data():
    data = pd.read_csv(data_mode_path)
    data_X = data.drop(['学号/工号', '姓名', 'y_label', '量表得分05', '量表得分06', '量表得分07', '量表得分11', '量表得分15', '性别'], axis=1)
    data_Y = data['y_label']

 #   data_X['性别'] = data_X['性别'].map({'男': 1, '女': 0})

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

    classification_report('XGB', Y_test, Y_pred)


def use_lr():
    X, Y = get_data()

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

    lr = LogisticRegression()
    lr.fit(X_train, Y_train)

    Y_pred = lr.predict(X_test)

    classification_report('Logistic Regression',Y_test, Y_pred)


def use_svc():
    X, Y = get_data()

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=5)

    svc = SVC(kernel='sigmoid')
    svc.fit(X_train, Y_train)

    Y_pred = svc.predict(X_test)

    classification_report('SVC', Y_test, Y_pred)

if __name__ == '__main__':
    print('begin')
    use_xgb()
