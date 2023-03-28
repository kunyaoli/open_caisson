from skmultilearn.adapt import MLkNN
from sklearn.model_selection import GridSearchCV
import pandas as pd
import sklearn.metrics as metrics
from sklearn.model_selection import KFold
import numpy as np
#from matplotlib import pyplot as plt
from sklearn import preprocessing
import pickle
import shap


def cal_acc_cls(cls, Y_test, predictions):
    cls = int(cls)
    acc = []
    for num in range(Y_test.shape[0]):
        result = Y_test.values[num].astype("int") + predictions.toarray()[num]
        # print(result)
        if cls == 1:
            if np.max(result) == 1:
                data_pred_cls = 0
            else:
                data_pred_cls = pd.Series(result).value_counts()[cls + 1]
        else:
            data_pred_cls = pd.Series(result).value_counts()[cls]
        # print("pred",data_pred_cls)
        data_true_cls = pd.Series(Y_test.values[num].astype("int")).value_counts()[cls]
        # print("True",data_true_cls)
        acc_num = np.round(data_pred_cls / data_true_cls, 4)
        acc.append(acc_num)
    return np.mean(acc)


# load data
X_train = pd.read_csv("../02data/Train_X_data.csv")
X_test = pd.read_csv("../02data/Test_X_data.csv")
Y_train = pd.read_csv("../02data/Train_Y_data.csv")
Y_test = pd.read_csv("../02data/Test_Y_data.csv")

X = pd.concat([X_train,X_test],axis=0)
Y = pd.concat([Y_train,Y_test],axis = 0)
Data = pd.concat([X,Y],axis=1)
#result = pd.DataFrame()
kf = KFold(n_splits=5, random_state=0)

num = 0
hamming_loss_l = []
acc_1_l = []
acc_0_l = []
tatal_acc = []
for train_index, test_index in kf.split(X):
    # print("TRAIN:", train_index, "TEST:", test_index)
    train_index = train_index[0:]
    print(len(test_index))
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    scaler = preprocessing.MinMaxScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    y_train, y_test = Y.iloc[train_index], Y.iloc[test_index]

    classifier = MLkNN(k=1)

    # train
    classifier.fit(X_train, y_train.values)

    # predict
    predictions = classifier.predict(X_test)
    predictions_proba = classifier.predict_proba(X_test)

    # metrcis
    # 1 hamming loss
    hamming_loss = metrics.hamming_loss(y_test.values, predictions.toarray())
    # 1 acc
    acc_1 = cal_acc_cls(1, y_test, predictions)
    # 0 acc
    cal_acc_cls(0, y_test, predictions)
    acc_0 = cal_acc_cls(0, y_test, predictions)

    acc = []
    for num in range(y_test.shape[0]):
        true_num = np.sum(y_test.values[num].astype("int") == predictions.toarray()[num])
        # print(true_num)
        acc_num = np.round(true_num / 38, 4)
        # print(acc_num)
        acc.append(acc_num)

    hamming_loss_l.append(hamming_loss)
    acc_0_l.append(acc_0)
    acc_1_l.append(acc_1)
    tatal_acc.append(np.mean(acc))
    print(
        "hamming loss:%6.3f---1 acc:%6.3f---0 acc:%6.3f---tatal acc:%6.3f" % (hamming_loss, acc_1, acc_0, np.mean(acc)))
    print(y_test.shape)
    result_true = pd.DataFrame(y_test)
    result_pred = pd.DataFrame(predictions.toarray())
    result_pred_proba = pd.DataFrame(predictions_proba.toarray())
    #result_true.to_csv('result_true.csv')
    #result_pred.to_csv("result_pred.csv")
    #result_pred_proba.to_csv("result_pred_proba.csv")

print("-*"*20)
print("hamming loss:%6.5f---1 acc:%6.5f---0 acc:%6.5f---tatal acc:%6.5f" %
      (np.mean(hamming_loss_l), np.mean(acc_1_l), np.mean(acc_0), np.mean(tatal_acc)))

with open('clf.pickle','wb') as f:
    pickle.dump(classifier,f) #将训练好的模型clf存储在变量f中，且保存到本地

