# Apply traditional stacking model to unbalanced data
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import time

data = pd.read_csv('/Users/yuxinma/Downloads/dataverse_files/standardized_data.csv')
y = data['y1']
x = data.iloc[:, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]

# Apply 5 classification algorithms
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,
                              GradientBoostingClassifier, ExtraTreesClassifier)
from sklearn.svm import SVC

rf_model = RandomForestClassifier()
adb_model = AdaBoostClassifier()
gdbc_model = GradientBoostingClassifier()
et_model = ExtraTreesClassifier()
svc_model = SVC()


def get_stacking(clf, x_train, y_train, x_test, n_folds=10):
    """
    This function is the core of stacking. It uses the cross-validation method to get the secondary training set.
    The values ​​of x_train, y_train, and x_test should be numpy.ndarray
    If the input is a pandas.DataFrame, an error is reported.
    """
    train_num, test_num = x_train.shape[0], x_test.shape[0]
    second_level_train_set = np.zeros((train_num,))
    second_level_test_set = np.zeros((test_num,))
    test_nfolds_sets = np.zeros((test_num, n_folds))
    kf = KFold(n_splits=n_folds)
    for i, (train_index, test_index) in enumerate(kf.split(x_train)):
        x_tra, y_tra = x_train[train_index], y_train[train_index]
        x_tst, y_tst = x_train[test_index], y_train[test_index]

        clf.fit(x_tra, y_tra)  # Model clf number i of K folds

        second_level_train_set[test_index] = clf.predict(x_tst)
        test_nfolds_sets[:, i] = clf.predict(x_test)  # bi

    second_level_test_set[:] = test_nfolds_sets.mean(axis=1)
    return second_level_train_set, second_level_test_set


chishu = 999
train_x, test_x, train_y, test_y = train_test_split(np.array(x), np.array(y), test_size=0.2)
train_sets = []
test_sets = []
start = time.clock()
for clf in [rf_model, adb_model, gdbc_model, et_model, svc_model]:
    train_set, test_set = get_stacking(clf, train_x, train_y, test_x)
    train_sets.append(train_set)  # test: A=[a1,a10] secondary input training
    test_sets.append(test_set)  # test: B secondary input testing
meta_train = np.concatenate([result_set.reshape(-1, 1) for result_set in train_sets], axis=1)  # A
meta_test = np.concatenate([y_test_set.reshape(-1, 1) for y_test_set in test_sets], axis=1)

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model.logistic import  LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score

# Logistic Regression
classifer = LogisticRegression()
classifer.fit(meta_train, train_y)
test_pre = classifer.predict(meta_test)
train_pre = classifer.predict(meta_train)

# Confusion Matrix
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
from sklearn.metrics import confusion_matrix

confusion_matrix = confusion_matrix(test_y, test_pre)
print(confusion_matrix)
plt.matshow(confusion_matrix)
plt.title('Confusion Matrix')
plt.colorbar()
plt.ylabel('True values')
plt.xlabel('Prediction values')
plt.show()
# plt.savefig('Confusion matrix'+str(cishu)+'.jpg')

# Compute accuracy
test_accuracy1 = (confusion_matrix[0][0] + confusion_matrix[1][1]) / sum(sum(confusion_matrix))
print('Accuracy:' + ' ' + str(test_accuracy1))

# Accuracy
print('Accuracy:' + ' ' + str(sum(test_pre == test_y) / len(test_y)))

# Fitted accuracy
train_accuracy1 = sum(train_pre == train_y) / len(train_y)
print('Fitted accuracy:' + ' ' + str(sum(train_pre == train_y) / len(train_y)))

# Precision & Recall
precision1 = confusion_matrix[1][1] / (confusion_matrix[1][1] + confusion_matrix[0][1])
recall1 = confusion_matrix[1][1] / (confusion_matrix[1][1] + confusion_matrix[1][0])
print('Precision:' + ' ' + str(precision1))
print('Recall:' + ' ' + str(recall1))
# Compute F-value
from sklearn.metrics import f1_score

print('f1 score :{:.2f}'.format(f1_score(test_y, test_pre)))

from sklearn.metrics import classification_report

print(classification_report(test_y, test_pre))

# ROC AUC
from sklearn.metrics import roc_curve, auc

# Plot curve
predictions = classifer.predict_proba(meta_test)
fpr1, tpr1, thresholds = roc_curve(test_y, predictions[:, 1])
roc_auc = auc(fpr1, tpr1)
plt.plot(fpr1, tpr1, 'b', label='AUC = %0.2f' % roc_auc)
plt.xlabel('FPR')
plt.ylabel('TPR(recall)')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.ylabel('Recall')
plt.xlabel('Fall-out')
plt.savefig('ROC' + str(chishu) + '.jpg')
plt.show()

end = time.clock()
print("The function run time is : %.03f seconds" % (end - start))



# Apply traditional stacking model to balanced data
data = pd.read_csv('/Users/yuxinma/Downloads/dataverse_files/smote_standardized_data.csv')
y = data['y1']
x = data.iloc[:, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]

rf_model = RandomForestClassifier()
adb_model = AdaBoostClassifier()
gdbc_model = GradientBoostingClassifier()
et_model = ExtraTreesClassifier()
svc_model = SVC()

chishu = 999
train_x, test_x, train_y, test_y = train_test_split(np.array(x), np.array(y), test_size=0.2)
train_sets = []
test_sets = []
start = time.clock()
for clf in [rf_model, adb_model, gdbc_model, et_model, svc_model]:
    train_set, test_set = get_stacking(clf, train_x, train_y, test_x)
    train_sets.append(train_set)
    test_sets.append(test_set)
meta_train = np.concatenate([result_set.reshape(-1, 1) for result_set in train_sets], axis=1)  # A
meta_test = np.concatenate([y_test_set.reshape(-1, 1) for y_test_set in test_sets], axis=1)


classifer = LogisticRegression()
classifer.fit(meta_train, train_y)
test_pre = classifer.predict(meta_test)
train_pre = classifer.predict(meta_train)


plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
from sklearn.metrics import confusion_matrix

confusion_matrix = confusion_matrix(test_y, test_pre)
print(confusion_matrix)
plt.matshow(confusion_matrix)
plt.title('Confusion Matrix')
plt.colorbar()
plt.ylabel('True values')
plt.xlabel('Prediction values')
plt.show()
# plt.savefig('Confusion matrix'+str(cishu)+'.jpg')


test_accuracy2 = sum(test_pre == test_y) / len(test_y)
print('Accuracy:' + ' ' + str(test_accuracy2))

print('Accuracy:' + ' ' + str(sum(test_pre == test_y) / len(test_y)))

train_accuracy2 = sum(train_pre == train_y) / len(train_y)
print('Fitted accuracy:' + ' ' + str(sum(train_pre == train_y) / len(train_y)))

precision2 = confusion_matrix[1][1] / (confusion_matrix[1][1] + confusion_matrix[0][1])
recall2 = confusion_matrix[1][1] / (confusion_matrix[1][1] + confusion_matrix[1][0])
print('Precision:' + ' ' + str(precision2))
print('Recall:' + ' ' + str(recall2))

print('f1 score :{:.2f}'.format(f1_score(test_y, test_pre)))

print(classification_report(test_y, test_pre))

predictions = classifer.predict_proba(meta_test)
fpr2, tpr2, thresholds = roc_curve(test_y, predictions[:, 1])
roc_auc = auc(fpr2, tpr2)
plt.plot(fpr2, tpr2, 'b', label='AUC = %0.2f' % roc_auc)
plt.xlabel('FPR')
plt.ylabel('TPR(recall)')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.ylabel('Recall')
plt.xlabel('Fall-out')
plt.savefig('ROC' + str(chishu) + '.jpg')
plt.show()

end = time.clock()
print("The function run time is : %.03f seconds" % (end - start))

# Improved accuracy stacking model to balanced data
rf_model = RandomForestClassifier()
adb_model = AdaBoostClassifier()
gdbc_model = GradientBoostingClassifier()
et_model = ExtraTreesClassifier()
svc_model = SVC()


def get_stacking(clf, x_train, y_train, x_test, n_folds=10):

    train_num, test_num = x_train.shape[0], x_test.shape[0]
    second_level_train_set = np.zeros((train_num,))
    second_level_test_set = np.zeros((test_num,))
    test_nfolds_sets = np.zeros((test_num, n_folds))
    kf = KFold(n_splits=n_folds)
    a = np.zeros(n_folds)
    for i, (train_index, test_index) in enumerate(kf.split(x_train)):
        x_tra, y_tra = x_train[train_index], y_train[train_index]
        x_tst, y_tst = x_train[test_index], y_train[test_index]
        clf.fit(x_tra, y_tra)
        second_level_train_set[test_index] = clf.predict(x_tst)
        a[i] = (sum(clf.predict(x_tst) == y_tst) / len(y_tst))
        test_nfolds_sets[:, i] = clf.predict(x_test)
    a = a / sum(a)  # Accuracy ratio
    second_level_test_set = sum([test_nfolds_sets[:, i] * a[i] for i in range(0, n_folds)])
    # second_level_test_set[:]
    return second_level_train_set, second_level_test_set


chishu = 999
train_x, test_x, train_y, test_y = train_test_split(np.array(x), np.array(y), test_size=0.2)
train_sets = []
test_sets = []
start = time.clock()
for clf in [rf_model, adb_model, gdbc_model, et_model, svc_model]:
    train_set, test_set = get_stacking(clf, train_x, train_y, test_x)
    train_sets.append(train_set)
    test_sets.append(test_set)
meta_train = np.concatenate([result_set.reshape(-1, 1) for result_set in train_sets], axis=1)
meta_test = np.concatenate([y_test_set.reshape(-1, 1) for y_test_set in test_sets], axis=1)


classifer = LogisticRegression()
classifer.fit(meta_train, train_y)
test_pre = classifer.predict(meta_test)
train_pre = classifer.predict(meta_train)


plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

confusion_matrix = confusion_matrix(test_y, test_pre)
print(confusion_matrix)
plt.matshow(confusion_matrix)
plt.title('Confusion Matrix')
plt.colorbar()
plt.ylabel('True values')
plt.xlabel('Prediction values')
plt.show()
# plt.savefig('Confusion matrix'+str(cishu)+'.jpg')


test_accuracy3 = (confusion_matrix[0][0] + confusion_matrix[1][1]) / sum(sum(confusion_matrix))
print('Accuracy:' + ' ' + str(test_accuracy3))

print('Accuracy:' + ' ' + str(sum(test_pre == test_y) / len(test_y)))

train_accuracy3 = sum(train_pre == train_y) / len(train_y)
print('Fitted accuracy:' + ' ' + str(sum(train_pre == train_y) / len(train_y)))

precision3 = confusion_matrix[1][1] / (confusion_matrix[1][1] + confusion_matrix[0][1])
recall3 = confusion_matrix[1][1] / (confusion_matrix[1][1] + confusion_matrix[1][0])
print('Precision:' + ' ' + str(precision3))
print('Recall:' + ' ' + str(recall3))


print('f1 score :{:.2f}'.format(f1_score(test_y, test_pre)))

print(classification_report(test_y, test_pre))

predictions = classifer.predict_proba(meta_test)
fpr3, tpr3, thresholds = roc_curve(test_y, predictions[:, 1])
roc_auc = auc(fpr3, tpr3)
plt.plot(fpr3, tpr3, 'b', label='AUC = %0.2f' % roc_auc)
plt.xlabel('FPR')
plt.ylabel('TPR(recall)')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.ylabel('Recall')
plt.xlabel('Fall-out')
plt.savefig('ROC_Titanic_accurate.jpg')
plt.show()

end = time.clock()
print("The function run time is : %.03f seconds" % (end - start))


# Improved stacking model based on voting accuracy

rf_model = RandomForestClassifier()
adb_model = AdaBoostClassifier()
gdbc_model = GradientBoostingClassifier()
et_model = ExtraTreesClassifier()
svc_model = SVC()

start = time.clock()
train_sets = []
test_sets = []
for clf in [rf_model, adb_model, gdbc_model, et_model, svc_model]:
    print(clf)
    train_set, test_set = get_stacking(clf, train_x, train_y, test_x)  #
    train_sets.append(train_set)
    test_sets.append(test_set)

meta_train = np.concatenate([result_set.reshape(-1, 1) for result_set in train_sets], axis=1)  # A
meta_test = np.concatenate([y_test_set.reshape(-1, 1) for y_test_set in test_sets], axis=1)

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.neural_network import MLPClassifier

clf1 = RandomForestClassifier()
clf2 = ExtraTreesClassifier()
clf3 = LogisticRegression()
eclf = VotingClassifier(estimators=[('dt', clf1), ('lr', clf2), ('MLP', clf3)], voting='soft', weights=[1, 1, 1])
# Control the weights of each algorithm, voting = ’soft' uses soft weights
clf1.fit(meta_train, train_y)
clf2.fit(meta_train, train_y)
clf3.fit(meta_train, train_y)
eclf.fit(meta_train, train_y)
test_pre = eclf.predict(meta_test)
train_pre = eclf.predict(meta_train)


plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

confusion_matrix = confusion_matrix(test_y, test_pre)
print(confusion_matrix)
plt.matshow(confusion_matrix)
plt.title('Confusion Matrix')
plt.colorbar()
plt.ylabel('True values')
plt.xlabel('Prediction values')
plt.show()

test_accuracy4 = (confusion_matrix[0][0] + confusion_matrix[1][1]) / sum(sum(confusion_matrix))
print('Accuracy:' + ' ' + str(test_accuracy4))

train_accuracy4 = sum(train_pre == train_y) / len(train_y)
print('Fitted Accuracy' + ' ' + str(sum(train_pre == train_y) / len(train_y)))

precision4 = confusion_matrix[1][1] / (confusion_matrix[1][1] + confusion_matrix[0][1])
recall4 = confusion_matrix[1][1] / (confusion_matrix[1][1] + confusion_matrix[1][0])
print('Precision:' + ' ' + str(precision4))
print('Recall:' + ' ' + str(recall4))


print('f1 score :{:.2f}'.format(f1_score(test_y, test_pre)))


print(classification_report(test_y, test_pre))


predictions = eclf.predict_proba(meta_test)
fpr4, tpr4, thresholds = roc_curve(test_y, predictions[:, 1])
roc_auc = auc(fpr4, tpr4)
plt.plot(fpr4, tpr4, 'b', label='AUC = %0.2f' % roc_auc)
plt.xlabel('FPR')
plt.ylabel('TPR(recall)')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.ylabel('Recall')
plt.xlabel('Fall-out')
plt.savefig('ROCStackingvote_data1:1.jpg')
# plt.show()

end = time.clock()
print("The function run time is : %.03f seconds" % (end - start))



# Voting and accuracy-based stacking models with 5 classifications
import pandas as pd
import numpy as np
import time
from pylab import *
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, train_test_split
from sklearn.tree import DecisionTreeClassifier

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

rf_model = RandomForestClassifier()
adb_model = AdaBoostClassifier()
gdbc_model = GradientBoostingClassifier()
et_model = ExtraTreesClassifier()
svc_model = SVC()


def get_stacking(clf, x_train, y_train, x_test, n_folds=10):

    train_num, test_num = x_train.shape[0], x_test.shape[0]
    second_level_train_set = np.zeros((train_num,))
    second_level_test_set = np.zeros((test_num,))
    test_nfolds_sets = np.zeros((test_num, n_folds))
    kf = KFold(n_splits=n_folds)
    a = np.zeros(n_folds)
    for i, (train_index, test_index) in enumerate(kf.split(x_train)):
        x_tra, y_tra = x_train[train_index], y_train[train_index]
        y_tra = np.array([y_tra[i][0] for i in range(0, len(y_tra))])
        x_tst, y_tst = x_train[test_index], y_train[test_index]
        y_tst = np.array([y_tst[i][0] for i in range(0, len(y_tst))])
        clf.fit(x_tra, y_tra)
        second_level_train_set[test_index] = clf.predict(x_tst)
        a[i] = (sum(clf.predict(x_tst) == y_tst) / len(y_tst))
        test_nfolds_sets[:, i] = clf.predict(x_test)
    a = a / sum(a)
    second_level_test_set = sum([test_nfolds_sets[:, i] * a[i] for i in range(0, n_folds)], axis=0)
    return second_level_train_set, a, second_level_test_set



names = y.columns
# for each in names[1:]:
#    y[each]=(y[each]-y[each].min())/(y[each].max()-y[each].min())  # Normalization

num_clusters = 5

length = []
data = y[names[1:]]
km_cluster = KMeans(n_clusters=num_clusters, max_iter=300, n_init=40, init='k-means++').fit(data)
t = km_cluster.cluster_centers_  # Centroids
y['labels'] = km_cluster.labels_  # Labels

y.target = y[['y1', 'labels']]
y.data = y[['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'labels']]
train_x, test_x, train_y, test_y = train_test_split(y.data, y.target, test_size=0.2)
all_data = dict()
for i in range(0, num_clusters):
    # print(i)
    data_small = dict()
    data_small[0] = np.array(
        train_x.loc[train_x['labels'] == i][['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10']])
    data_small[1] = np.array(
        test_x.loc[test_x['labels'] == i][['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10']])
    data_small[2] = np.array(train_y.loc[train_y['labels'] == i][['y1']])
    data_small[3] = np.array(test_y.loc[test_y['labels'] == i][['y1']])
    all_data[i] = data_small
    length.append(sum(test_x['labels'] == i))

result_train = dict()
result_test = dict()
s_all = 0

start = time.clock()
cluster_test_accuracy = []
cluster_train_accuracy = []
cluster_precision = []
cluster_recall = []
cluster_fpr = []
cluster_tpr = []
for k in range(0, num_clusters):
    train_x = all_data[k][0]
    test_x = all_data[k][1]
    train_y = all_data[k][2]
    test_y = all_data[k][3]

    train_sets = []
    test_sets = []
    for clf in [rf_model, adb_model, gdbc_model, et_model, svc_model]:
        train_set, a, test_set = get_stacking(clf, train_x, train_y, test_x)  #
        train_sets.append(train_set)
        test_sets.append(test_set)
    meta_train = np.concatenate([result_set.reshape(-1, 1) for result_set in train_sets], axis=1)  # A
    meta_test = np.concatenate([y_test_set.reshape(-1, 1) for y_test_set in test_sets], axis=1)  # B


    train_y = np.array([train_y[i][0] for i in range(0, len(train_y))])
    test_y = np.array([test_y[i][0] for i in range(0, len(test_y))])


    clf1 = RandomForestClassifier()
    clf2 = LogisticRegression()
    clf3 = ExtraTreesClassifier()
    eclf = VotingClassifier(estimators=[('dt', clf1), ('lr', clf2), ('MLP', clf3)], voting='soft',
                            weights=[1, 1, 1])
    clf1.fit(meta_train, train_y)
    clf2.fit(meta_train, train_y)
    clf3.fit(meta_train, train_y)
    eclf.fit(meta_train, train_y)
    test_pre = eclf.predict(meta_test)
    train_pre = eclf.predict(meta_train)


    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    confusion_matrix = confusion_matrix(test_y, test_pre)
    print(confusion_matrix)
    plt.matshow(confusion_matrix)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.ylabel('True values')
    plt.xlabel('Prediction values')
    plt.show()
    plt.savefig('Confusion Matrix with 5 classifications' + str(k) + '.jpg')

    test_accuracy = (confusion_matrix[0][0] + confusion_matrix[1][1]) / sum(sum(confusion_matrix))
    print('Accuracy:' + ' ' + str(test_accuracy))

    train_accuracy = sum(train_pre == train_y) / len(train_y)
    print('Fitted accuracy:' + ' ' + str(sum(train_pre == train_y) / len(train_y)))
    # s_all = s_all + sum(train_pre ==train_y) + sum(test_pre ==test_y)

    precision = confusion_matrix[1][1] / (confusion_matrix[1][1] + confusion_matrix[0][1])
    recall = confusion_matrix[1][1] / (confusion_matrix[1][1] + confusion_matrix[1][0])
    print('Precision:' + ' ' + str(precision))
    print('Recall:' + ' ' + str(recall))
    cluster_test_accuracy.append(test_accuracy)
    cluster_train_accuracy.append(train_accuracy)
    cluster_precision.append(precision)
    cluster_recall.append(recall)

    print('f1 score :{:.2f}'.format(f1_score(test_y, test_pre)))

    print(classification_report(test_y, test_pre))

    predictions = eclf.predict_proba(meta_test)
    fpr, tpr, thresholds = roc_curve(test_y, predictions[:, 1])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.xlabel('FPR')
    plt.ylabel('TPR(recall)')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.ylabel('Recall')
    plt.xlabel('Fall-out')
    plt.savefig('ROC351_7vote_0312' + str(k) + '.jpg')  # +str(cishu)
    plt.show()
    cluster_fpr.append(fpr)
    cluster_tpr.append(tpr)

end = time.clock()
print("The function run time is : %.03f seconds" % (end - start))



# Model evaluation results after calculating the weighted average
# after classification (fitted accuracy, test accuracy, precision, recall)
final_test_accuracy = sum((length / (sum(length))) * cluster_test_accuracy)
final_train_accuracy = sum((length / (sum(length))) * cluster_train_accuracy)
final_cluster_precision = sum((length / (sum(length))) * cluster_precision)
final_cluster_recall = sum((length / (sum(length))) * cluster_recall)
print(final_test_accuracy)
print(fianl_train_accuracy)
print(final_cluster_precision)
print(final_cluster_recall)

# Comparison of ROC curves between balanced data of traditional and
# improved-accuracy-based stacking models

# Plots
plt.title('Receiver Operating Characteristic')
roc_auc31 = auc(fpr3, tpr3)
roc_auc21 = auc(fpr2, tpr2)
plt.plot(fpr2, tpr2, 'green', label='traditional stacking：AUC = %0.2f' % roc_auc21)
plt.plot(fpr3, tpr3, 'lightcoral', label='accuracy-based stacking：AUC = %0.2f' % roc_auc31)
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.legend(loc='lower right')
plt.xlabel('Recall')
plt.ylabel('Fall-out')
plt.show()

# Fitted & test accuracy
# encoding=utf-8

from pylab import *

mpl.rcParams['font.sans-serif'] = ['SimHei']
x = [1, 2, 3, 4, 5]
train_accuracy = [train_accuracy1, train_accuracy2, train_accuracy3, train_accuracy4, final_train_accuracy]
test_accuracy = [test_accuracy1, test_accuracy2, test_accuracy3, test_accuracy4, final_test_accuracy]
precision = [precision1, precision2, precision3, precision4, final_cluster_precision]
recall = [recall1, recall2, recall3, recall4, final_cluster_recall]
plt.plot(x, train_accuracy, marker='v', mec='g', label='Fitted accuracy')
plt.plot(x, test_accuracy, marker='+', mec='y', label='test accuracy')
plt.legend()
plt.xlim([0.0, 5.1])
plt.ylim([0.93, 0.95])
plt.xlabel('Models')
plt.ylabel('Ratios')
plt.show()


# Precision & Recall
# encoding=utf-8

mpl.rcParams['font.sans-serif'] = ['SimHei']
x = [1, 2, 3, 4, 5]

plt.plot(x, recall, marker='v', mec='g', label='Recall')
plt.plot(x, precision, marker='+', mec='y', label='Precision')
plt.legend()
plt.xlim([0.0, 5.0])
plt.ylim([0, 1])
plt.xlabel('Models')
plt.ylabel('Ratios')
plt.show()

# Unbalanced vs balanced data stacking models comparison
# encoding=utf-8
mpl.rcParams['font.sans-serif'] = ['SimHei']
x = [1, 2, 3, 4]
tradition = [test_accuracy1, train_accuracy1, precision1, recall1]
balanced_tradition = [test_accuracy2, train_accuracy2, precision2, recall2]
recall = [recall1, recall2, recall3, recall4, final_cluster_recall]
plt.plot(x, tradition, marker='v', mec='g', label='Unbalanced')
plt.plot(x, balanced_tradition, marker='+', mec='y', label='Balanced')
plt.legend()
plt.xlim([0.0, 5.0])
plt.ylim([0, 1])
plt.xlabel('Models')
plt.ylabel('Ratios')
plt.show()

# Unbalanced vs balanced data stacking models comparison of ROC & AUC
# encoding=utf-8

roc_auc11 = auc(fpr1, tpr1)
roc_auc21 = auc(fpr2, tpr2)
plt.plot(fpr1, tpr1, 'greenyellow', label='Unbalanced: AUC = %0.2f' % roc_auc11)
plt.plot(fpr2, tpr2, 'lightcoral', label='Balanced: AUC = %0.2f' % roc_auc21)
plt.xlabel('FPR')
plt.ylabel('TPR(recall)')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.ylabel('Recall')
plt.xlabel('Fall-out')
plt.savefig('ROC comparison.jpg')
plt.show()


# ROC comparison among models

plt.title('Receiver Operating Characteristic')
plt.plot(fpr1, tpr1, color='green', label='Unbalanced stacking')
plt.plot(fpr2, tpr2, color='red', label='Balanced stacking')
plt.plot(fpr3, tpr3, color='skyblue', label='Balanced accurate-stacking')
plt.plot(fpr4, tpr4, color='blue', label='Balanced vote-accurate-stacking')
plt.plot(fpr5, tpr5, color='blue', label='Balanced based on classifications vote-accurate-stacking')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.legend(loc='lower right')
plt.xlabel('Recall')
plt.ylabel('Fall-out')
plt.show()

# Based on 5 classifications accurate_and_voting_stacking


mpl.rcParams['font.sans-serif'] = ['SimHei']
x = [1, 2, 3, 4, 5]
cluster_test_accuracy = [0.976, 0.95, 0.96214967, 0.88534, 0.9]
cluster_train_accuracy = [0.975249456, 0.9487, 0.962737422, 0.866696894, 0.89534558]
cluster_precision = [0.98, 0.924242, 0.896853, 0.88, 0.89858]
cluster_recall = [0.487562, 0.493927, 0.48918, 0.6743, 0.674277]
plt.title('Comparison of model evaluation indicators for five types of classifications')
plt.plot(x, cluster_test_accuracy, marker='+', color='green', label='Test accuracy')
plt.plot(x, cluster_train_accuracy, marker='o', color='red', label='Train accuracy')
plt.plot(x, cluster_precision, marker='v', color='skyblue', label='Precision')
plt.plot(x, cluster_recall, marker='.', color='blue', label='Recall')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([1.0, 6.0])
plt.ylim([0.45, 1.0])
plt.legend(loc='lower right')
plt.xlabel('Classifications')
plt.ylabel('Ratios')
plt.show()

# ROC Curves
plt.plot(cluster_fpr[0], cluster_tpr[0], 'orange', label='1st: AUC = %0.2f' % roc_auc[0])
plt.plot(cluster_fpr[1], cluster_tpr[1], 'red', label='2nd: AUC = %0.2f' % roc_auc[1])
plt.plot(cluster_fpr[2], cluster_tpr[2], 'green', label='3rd: AUC = %0.2f' % roc_auc[2])
plt.plot(cluster_fpr[3], cluster_tpr[3], 'blue', label='4th: AUC = %0.2f' % roc_auc[3])
plt.plot(cluster_fpr[4], cluster_tpr[4], 'yellow', label='5th: AUC = %0.2f' % roc_auc[4])
plt.xlabel('FPR')
plt.ylabel('TPR(recall)')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.ylabel('Recall')
plt.xlabel('Fall-out')
plt.savefig('ROC351_7vote_0314.jpg')  # +str(cishu)
plt.show()
