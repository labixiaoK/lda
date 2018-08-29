#encoding=utf-8
import json
import sys
import re
import os
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.externals import joblib
import jieba
import time
#load vocab
print 'load vocab'
dtf = open('data/dict/all.dict', 'r')
vocab = json.load(dtf)
dtf.close()

print 'load alltrain.clean'
sys.stdout.flush()
train_cont = []
train_label = []
with open('data/alltrain.clean', 'r') as fin:
    for line in fin:
        line = line.strip().split('\t')
        sid = line[0]
        label = line[1]
        ct = line[2].strip()
        train_label.append([sid, label])
        train_cont.append(ct)

print 'write train_label'
with open('train_label', 'w') as fo:
    for i in train_label:
        fo.write(i[0]+'\t'+i[1]+'\n')
sys.stdout.flush()
print 'load alltest.clean'
test_cont = []
test_label = []
with open('data/alltest.clean', 'r') as fin:
    for line in fin:
        line = line.strip().split('\t')
        sid = line[0]
        label = line[1]
        ct = line[2].strip()
        test_label.append([sid, label])
        test_cont.append(ct)

print 'write test_lael'
with open('test_label', 'w') as fo:
    for i in test_label:
        fo.write(i[0]+'\t'+i[1]+'\n')
print 'vectorizer'
sys.stdout.flush()
vec_st_time = time.time()
vectorizer = CountVectorizer(analyzer='word', vocabulary=vocab, tokenizer=jieba._lcut_all, lowercase=False)
X_train = vectorizer.fit_transform(train_cont)
X_test = vectorizer.transform(test_cont)
vec_end_time = time.time()

print 'vectorizer cost time(s): %d', (vec_end_time-vec_st_time)

print 'save train&test vectorizer'
np.save('X_train_ctvec', X_train)
np.save('X_test_ctvec', X_test)

print 'lda start'
sys.stdout.flush()
lda_st_time = time.time()

lda = LatentDirichletAllocation(n_topics=500, random_state=22, learning_method='online')
X_train_tps = lda.fit_transform(X_train)
X_test_tps = lda.transform(X_test)

print 'save lda'
np.save('X_train_tps', X_train_tps)
np.save('X_test_tps', X_test_tps)

joblib.dump(lda, 'lda.pkl')

lda_end_time = time.time()
print 'lda cost time(mins): %d', (lda_end_time-lda_st_time)/60.0

