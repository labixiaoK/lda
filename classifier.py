#encoding=utf-8
import json
import numpy as np
import re
import sys
import numpy as np
from sklearn.externals import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn import svm
from sklearn.linear_model import LogisticRegression

import jieba

import time


print 'labelencoder label'
sys.stdout.flush()
y_train = []
y_test = []

with open('train_label', 'r') as fin:
    for i in fin:
        y = i.strip().split('\t')[1]
        y_train.append(y)

with open('test_label', 'r') as fin:
    for i in fin:
        y = i.strip().split('\t')[1]
        y_test.append(y)

le = LabelEncoder()
y_train = le.fit_transform(y_train)
assert le.classes_.tolist() == ['1', '7', '8', '9']
y_test = le.transform(y_test)

print 'save y_train and y_test'
sys.stdout.flush()

np.save('y_train', y_train)
np.save('y_test', y_test)

print 'load X'
sys.stdout.flush()

X_train = np.load('X_train_tps.npy')
X_test = np.load('X_test_tps.npy')

print X_train.shape
print X_test.shape

print 'linear svc fit'
svc_st_time = time.time()
sys.stdout.flush()

svc = svm.LinearSVC(random_state=22)
svc.fit(X_train, y_train)
svc_pre = svc.predict(X_test)

svc_end_time = time.time()
print 'svc cost time/min: ', (svc_end_time-svc_st_time)/60
print 'save svc_pre and svc.pkl'
sys.stdout.flush()

joblib.dump(svc, 'svc.pkl')
np.save('svc_pre', svc_pre)

print 'logist rg fit'
lgt_st_time = time.time()
sys.stdout.flush()

lgt = LogisticRegression(solver='newton-cg', multi_class='multinomial', random_state=22)
lgt.fit(X_train, y_train)
lgt_pre = lgt.predict(X_test)

lgt_end_time = time.time()
print 'lgt cost time/min: ', (lgt_end_time-lgt_st_time)/60.0
print 'save lgt_pre and lgt.pkl'
sys.stdout.flush()

np.save('lgt_pre', lgt_pre)
joblib.dump(lgt, 'lgt.pkl')



print 'logist rg(ovr) fit'
lgt_st_time_ovr = time.time()
sys.stdout.flush()

lgt_ovr = LogisticRegression(solver='sag', multi_class='ovr', random_state=22)
lgt_ovr.fit(X_train, y_train)
lgt_ovr_pre = lgt_ovr.predict(X_test)

lgt_end_time_ovr = time.time()
print 'lgt_ovr cost time/min: ', (lgt_end_time_ovr-lgt_st_time_ovr)/60.0
print 'save lgt_ovr_pre and lgt_ovr.pkl'
sys.stdout.flush()

np.save('lgt_ovr_pre', lgt_ovr_pre)
joblib.dump(lgt_ovr, 'lgt_ovr.pkl')


print 'logist rg(l1) fit'
lgt_st_time_l1 = time.time()
sys.stdout.flush()

lgt_l1 = LogisticRegression(penalty='l1', solver='liblinear', multi_class='ovr', random_state=22)
lgt_l1.fit(X_train, y_train)
lgt_l1_pre = lgt_l1.predict(X_test)

lgt_end_time_l1 = time.time()
print 'lgt_l1 cost time/min: ', (lgt_end_time_l1-lgt_st_time_l1)/60.0
print 'save lgt_l1_pre and lgt_l1.pkl'
sys.stdout.flush()

np.save('lgt_l1_pre', lgt_l1_pre)
joblib.dump(lgt_l1, 'lgt_l1.pkl')




