#!/usr/bin/env python
# encoding: utf-8
# @author: zxding
# email: d.z.x@qq.com

from __future__ import print_function
import numpy as np
import sys

#将标签转换为onehot表示
def change_y_to_onehot(y, n_class = 5):
    onehot = []
    for label in y:
        tmp = [0] * n_class
        tmp[label-1] = 1
        onehot.append(tmp)
    return np.asarray(onehot, dtype=np.int32)

#读入文档
def load_inputs_document(input_file, word_id_file, max_sen_len, max_doc_len, encoding='utf8'):
    if type(word_id_file) is str:
        word_to_id = load_word_id_mapping(word_id_file)
    else:
        word_to_id = word_id_file

    x, y, sen_len, doc_len = [], [], [], []
    print('loading input {}...'.format(input_file))
    for line in open(input_file):
        line = line.lower().decode('utf8', 'ignore').split('\t\t')
        y.append(int(line[0]))

        t_sen_len = [0] * max_doc_len
        t_x = np.zeros((max_doc_len, max_sen_len), dtype=np.int)
        doc = ' '.join(line[1:])
        sentences = doc.split('<sssss>')
        i = 0
        for sentence in sentences:
            j = 0
            for word in sentence.split():
                if j < max_sen_len:
                    if word in word_to_id:
                        t_x[i, j] = word_to_id[word]
                        j += 1
                else:
                    break
            t_sen_len[i] = j
            i += 1
            if i >= max_doc_len:
                break

        doc_len.append(i)
        sen_len.append(t_sen_len)
        x.append(t_x)

    y = change_y_to_onehot(y)
    print('done!')

    return np.asarray(x), np.asarray(y), np.asarray(sen_len), np.asarray(doc_len)

def load_data_for_DSCNN_sen(input_file, word_to_id, max_doc_len, n_class=2, n_fold=0, index=1):
    x1, y1, doc_len1 = [], [], []
    x2, y2, doc_len2 = [], [], []

    print('loading input {}...'.format(input_file))
    num_truncated=0
    for line in open(input_file):
        line = line.split('\t\t')
        wordlist = line[-1].split()
        
        tmp_x = np.zeros((max_doc_len), dtype=np.int)
        i = 0
        for word in wordlist:
            if i >= max_doc_len:
                num_truncated+=1
                break
            if word in word_to_id:
                tmp_x[i] = word_to_id[word]
                i += 1
        
        tmp_y = np.zeros((n_class), dtype=np.int)
        tmp_y[int(line[0])%n_class]=1

        if n_fold and index % n_fold==0:
            x, y, doc_len = x2, y2, doc_len2
        else :
            x, y, doc_len = x1, y1, doc_len1
        x.append(tmp_x)
        y.append(tmp_y)
        doc_len.append(i)
        index+=1
    
    print('done!\ntruncated_Samples:',num_truncated)
    if n_fold:
        return np.asarray(x1), np.asarray(y1), np.asarray(doc_len1), np.asarray(x2), np.asarray(y2), np.asarray(doc_len2)
    else :
        return np.asarray(x1), np.asarray(y1), np.asarray(doc_len1)

def read_input_Data(word_to_id, max_doc_len, reverse_x = 0):
    x, y, doc_len = [], [], []
    print('sentence> ',end='')
    line = sys.stdin.readline()[:-1]
    tmp_x = np.zeros((max_doc_len), dtype=np.int)
    i = 0
    show=[]
    sentence = line.split()
    if reverse_x:
        sentence.reverse()
    for word in sentence:
        if i >= max_doc_len:
            print('truncated')
            break
        if word in word_to_id:
            show.append(word)
            tmp_x[i] = word_to_id[word]
            i += 1
        else:
            show.append('UNK')
    
    tmp_y = np.array([1,0], dtype=np.int)
    x.append(tmp_x)
    y.append(tmp_y)
    doc_len.append(i)
    print('sentence>',' '.join(show))
    # print('x {} y {} doc_len {}'.format(x,y,doc_len))
    return np.asarray(x), np.asarray(y), np.asarray(doc_len)
# read_input_Data({'dzx':1,'is':2,'handsome':3}, 50)
