#landmark-manage.py
import pandas as pd
def generate_and_save_train_Xy(labelmaps,train): #train : DataFrame file of train.csv
	#X[i]['train'] : 画像ファイルのid番号 X[i]['result'] : ラベル、1または-1
	for i in len(labelmaps):
		for j in len(train):
			X[i] = pd.DataFrame({'train':[],'result':[]})
			if(train['landmark_id'][j] in labelmaps[i][1]):
				X[i] = X[i].append(pd.Series([train['id'][j],1],index=X[i].columns),ignore_index=True)
			else:
				X[i] = X[i].append(pd.Series([train['id'][j],-1],index=X[i].columns),ignore_index=True)
		X[i].to_csv("Xy"+str(i)+".csv",index=False,columns=['train','result'])

def preprocess(t):
	
	return t

def train_all_classifiers(clf,labelmaps):
	clf_all = []
	for i in len(labelmaps):
		clf.clear()
		data = pd.read_csv('Xy'+str(i)+'.csv')
		X = image_process(data['train'])
		y = data['result'].astype('int32').to_numpy()
		clf.fit(X,y)
		clf_all.append(clf.dup())
	return clf_all

if __name__ == '__main__':
	train = pd.read_csv('train.csv')
	test = pd.read_csv('test.csv')

	train = preprocess(train)
	x_train = train.drop(['url','landmark_id'],axis=1).id.apply(str) #only id
	t_train = train['landmark_id'].astype('int32').to_numpy()

	x_test = test['id'].tolist()

	from sklearn.model_selection import train_test_split

	x_t,x_v,t_t,t_v = train_test_split(x_train,t_train,test_size=0.3,random_state=20190520)

	print("Finished")

"""
#read neural network properties (epoch,batchsize)

file_nn_properties = open("nn_properties.txt",'r')

lines = file_nn_properties.readlines()
for line in lines:
	print(line)

file_nn_properties.close()

optimizer = chainer.optimizers.SGD(lr=0.03)
n_epoch = 2000
n_batchsize = 30


#read_end
import landmark_nn
net = landmark_nn.Net()
predictor = landmark_nn.MyNN(optimizer,net,n_epoch,n_batchsize)
predictor.fit(x_t,x_v,t_t,t_v)

"""