#landmark-main.py

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import unit_classifier #分類機
import os
from multiprocessing import Process
import random
from PIL import Image
import keras
from keras.models import load_model
import compute_final_output
LABELMAP_FINISHED = 0
LABELMAP_NUMBER = 16-LABELMAP_FINISHED
import sys
def printProgress (iteration, total, prefix = '', suffix = '', decimals = 1, barLength = 100):
	formatStr = "{0:." + str(decimals) + "f}"
	percent = formatStr.format(100 * (iteration / float(total)))
	filledLength = int(round(barLength * iteration / float(total)))
	bar = '#' * filledLength + '-' * (barLength - filledLength)
	sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percent, '%', suffix)),
	if iteration == total:
		sys.stdout.write('\n')
	sys.stdout.flush()

def flatten(nested_list):
	return [e for inner_list in nested_list for e in inner_list]
"""
def generate_labelmaps(length):
	t = train.drop(['id','url','path'],axis=1)
	train_list = list(set(flatten(t.values.tolist())))
	labelmaps = []
	for i in range(length):
		printProgress(i, length, 'Progress:', 'Complete', 1, 50)
		random.shuffle(train_list)
		labelmaps.append([train_list[:round(len(train_list)/2)],train_list[round(len(train_list)/2):]])
	return labelmaps
"""
def generate_labelmaps(length):
	t = train.drop(['id','url','path'],axis=1)
	train_list = list(set(flatten(t.values.tolist())))
	train_list = [int(i) for i in train_list]
	max_map_id = max(train_list)
	labelmaps = [[None for x in range(max_map_id+1)] for y in range(length)]
	for i in range(length):
		random.shuffle(train_list)
		for j in train_list[:round(len(train_list)/2)]:
			labelmaps[i][int(j)] = 1
		for j in train_list[round(len(train_list)/2):]:
			labelmaps[i][int(j)] = -1
	return labelmaps

def generate_and_save_train_Xy(label,index,train): #train : DataFrame file of train.csv
	#X[i]['train'] : 画像ファイル名 X[i]['result'] : ラベル、1または-1
	print(str(os.getpid())+" Started")
	X = pd.DataFrame({'train':[],'result':[]})
	for j in range(len(train.index)):
		if(j%10000 == 9999):
			print(" Index : "+str(index)+" "+str(j)+" : Finished")
		if(label[int(train['landmark_id'][j])] == 1): #labelmaps[i][1]にj番目のデータのlandmark_idが含まれているのかを確認する
			X = X.append({'train': train['id'][j] ,'result':1},ignore_index=True)
		else: #含まれていない場合はlabelmaps[i][-1]に含まれている
			X = X.append({'train': train['id'][j] ,'result':0},ignore_index=True)
	X.to_csv("../data/Xy"+str(index+LABELMAP_FINISHED)+".csv",index=False,columns=['train','result'])

def image_process(filename_list,width=320,height=240):
	rgb_list = []
	cnt = 0
	length = len(filename_list)
	for filename in filename_list:
		printProgress(cnt, length, 'Progress:', 'Complete', 1, 50)
		file = Image.open("../data_train/"+filename+".jpg")
		file = file.resize((width,height),Image.ANTIALIAS)
		rgb_list.append(np.asarray(file)/255)
		cnt +=1
	return rgb_list

def train_all_classifiers(clf): 
	clf_all = [] #fit後の分類機の配列
	for i in range(LABELMAP_NUMBER):
		clf.clear()
		data = pd.read_csv('../data/Xy'+str(i)+'.csv')
		for j in range(int(len(data.index)/2000)):
			print("Currently in Classifier #%d , Data #%d" %(i,j*2000))
			data_divided = data.drop(data.index[:j*2000]).drop(data.index[(j+1)*2000:])
			X = np.array(image_process(data_divided['train'].to_numpy()))#image_process()はファイル名を読み、3-D array (2-D array * 3 (R,G,B))に変換する
			y = data_divided['result'].astype('int32').to_numpy()#1または-1
			y = keras.utils.to_categorical((y+1)/2)
			for k in range(len(y)):
				y[k] = y[k].reshape(1,-1)
			clf.fit(X,y)
		clf_all.append(clf.dup())
		clf.model.save("../models/clf"+str(i)+".h5")
	return clf_all
"""
def compute_final_output(clf_all,labelmaps):
	t = train.drop(['id','url','path'],axis=1)
	train_list = list(set(flatten(t.values.tolist())))
	train_list = [int(i) for i in train_list]
	max_map_id = max(train_list)
	y_score = [0]*max_map_id
	Xhat = np.array(image_process(test['id'].to_numpy()))
	for clf_ in clf_all:
		yhat = clf_.predict(Xhat)
"""
def save_model(clf_all):
	index = 0
	for clf in clf_all:
		clf.model.save("../models/clf"+str(index)+".h5")
		index += 1
if __name__ == '__main__':
	#train = pd.read_csv("../data/train.csv")
	#t = train.drop(['id','url','path'],axis=1)
	#train_list = list(set(flatten(t.values.tolist())))
	#train_list = [int(i) for i in train_list]
	#max_map_id = max(train_list)
	#test = pd.read_csv("../data/test.csv")
	"""
	#preprocessing
	check_path_train = lambda x : os.path.exists("../data_train/"+x+".jpg")
	check_path_test = lambda x : os.path.exists("../data_test/"+x+".jpg")
	train['path'] = train['id'].apply(check_path_train)
	test['path'] = test['id'].apply(check_path_test)
	train.drop(train.index[train.landmark_id == 'None'], inplace=True)
	train.drop(train.index[train.path == False],inplace=True)
	test.drop(test.index[test.path == False],inplace=True)
	train = train.reset_index(drop=True)
	test = test.reset_index(drop=True)
	#train = train.drop(train.index[100000:])
	labelmaps = generate_labelmaps(LABELMAP_NUMBER)
	procs = []
	for i in range(int(len(labelmaps)/16)):
		for j in range(16):
			proc = Process(target=generate_and_save_train_Xy,args=(labelmaps[i*16+j],i*16+j,train,))
			procs.append(proc)
			proc.start()
		for j in range(16):
			procs[i*16+j].join()
	"""
	#processing classification
	#clf = unit_classifier.Myclassifier(hight = 240, width = 320)

	#clf_all = train_all_classifiers(clf)	

	#save_model(clf_all)
	#postprocessing -> get results
	clf = unit_classifier.Myclassifier(hight = 240, width = 320)
	yhat = compute_final_output.compute_final_output(clf, LABELMAP_NUMBER)

	test['landmark'] = yhat
	test.to_csv("../data/output.csv", index=False, columns=['id','landmark'])