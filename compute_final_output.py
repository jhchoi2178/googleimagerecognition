# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.models import load_model
from PIL import Image
def flatten(nested_list):
    return [e for inner_list in nested_list for e in inner_list]
def image_process(filename,width=320,height=240):
    file = Image.open("../data_test/"+filename+".jpg")
    file = file.resize((width,height),Image.ANTIALIAS)
    return np.asarray(file)/255

def load_X(task_id=None, url_to_2darray=None):
    X = []
    t = pd.read_csv("../data/test.csv")
    for i in range(len(t.index)):
        X.append(t['id'][i])
    return X
def update(ydict, y, labelmaps,max_map_id):
    # y = 16개의 결과 (0 or 1)
    result = [0 for _ in range(max_map_id+1)]
    max_score = 0
    for task_id in range(len(y)):
        for landmark_id in range(len(labelmaps[task_id])):
            if(labelmaps[task_id]['result'][landmark_id]==y[task_id]):
                result[landmark_id] += 1
    for landmark_id in range(len(result)):
        if(max_score<result[landmark_id]):
            max_score = result[landmark_id]
            max_landmark_id = landmark_id
    max_score /= len(labelmaps)
    ydict = {"id": max_landmark_id,"score":max_score}
def normalize(yi):
    M = np.max(yi.values())
    m = sorted(yi.values(),reverse = True)
    ma = 0
    for i in range(0,10):
        ma += m[i]
    yi[label] = M / ma
def tostring(y_list):
    for y in y_list:
        s = ""
        for k,v in y.items():
            s = s + str(k) + " " + str(v)
        y = s
    return y_list
def trim(y_final):
    return tostring(y_final)
    
def compute_final_output(clf, LABELMAP_NUMBER):
    train = pd.read_csv("../data/train.csv")
    check_path_train = lambda x : os.path.exists("../data_train/"+x+".jpg")
    train['path'] = train['id'].apply(check_path_train)
    train.drop(train.index[train.landmark_id == 'None'], inplace=True)
    train.drop(train.index[train.path == False],inplace=True)
    train = train.reset_index(drop=True)
    t = train.drop(['id','url','path'],axis=1)
    train_list = list(set(flatten(t.values.tolist())))
    train_list = [int(i) for i in train_list]
    max_map_id = max(train_list)

    X = load_X()
    y_final = [0 for _ in range(len(X))]#y_final
    labelmaps = []
    clf_all = []
    for task_id in range(LABELMAP_NUMBER):
        labelmap = pd.read_csv('../data/Xy'+str(task_id)+'.csv')
        clf.clear()
        clf.model = load_model('../models/clf%s.h5'%task_id)
        clf_all.append(clf)
        labelmaps.append(labelmap)
    for i in range(len(X)):
        y = [0 for _ in range(LABELMAP_NUMBER)]
        if(i%1000 == 999):
            print('# %s Started'%i)
        for task_id in range(LABELMAP_NUMBER):
            print(task_id)
            y[task_id] = clf_all[task_id].predict(np.expand_dims(image_process(X[i]),axis=0)) #0~1
        update(y_final[i],y,labelmaps,max_map_id)
    return trim(y_final)

