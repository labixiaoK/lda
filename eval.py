from sklearn.metrics import accuracy_score, auc, f1_score, confusion_matrix, classification_report
import sys
import numpy as np

y_label = np.load('y_test.npy')
svc_pre = np.load('svc_pre.npy')
lgt_pre = np.load('lgt_pre.npy')
lgt_ovr_pre = np.load('lgt_ovr_pre.npy')
lgt_l1_pre = np.load('lgt_l1_pre.npy')



target_names = ['chinese', 'politics', 'history', 'geography']

print 'svc acc', accuracy_score(y_label, svc_pre) 
#print 'svc auc', auc(y_label, svc_pre) 
print 'svc f1', f1_score(y_label, svc_pre, average='micro')
print 'svc confusion matrix\n', confusion_matrix(y_label, svc_pre)
print 'svc report:\n', classification_report(y_label, svc_pre, target_names=target_names)

print 'lgt acc', accuracy_score(y_label, lgt_pre) 
#print 'lgt auc', auc(y_label, lgt_pre) 
print 'lgt f1', f1_score(y_label, lgt_pre, average='micro')
print 'lgt confusion matrix\n', confusion_matrix(y_label, lgt_pre) 
print 'lgt report:\n', classification_report(y_label, lgt_pre, target_names=target_names)


print 'lgt_ovr acc', accuracy_score(y_label, lgt_ovr_pre) 
print 'lgt_ovr f1', f1_score(y_label, lgt_ovr_pre, average='micro')
print 'lgt_ovr confusion matrix\n', confusion_matrix(y_label, lgt_ovr_pre) 
print 'lgt_ovr report:\n', classification_report(y_label, lgt_ovr_pre, target_names=target_names)


print 'lgt_l1 acc', accuracy_score(y_label, lgt_l1_pre) 
print 'lgt_l1 f1', f1_score(y_label, lgt_l1_pre, average='micro')
print 'lgt_l1 confusion matrix\n', confusion_matrix(y_label, lgt_l1_pre) 
print 'lgt_l1 report:\n', classification_report(y_label, lgt_l1_pre, target_names=target_names)


