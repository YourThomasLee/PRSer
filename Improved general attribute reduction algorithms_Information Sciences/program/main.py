#-*- encoding: UTF-8 -*-
from reducer import *
from preprocess import *
from granularity import *
import time
import os
import pandas as pd
import time, timeit
#shuttle(int),pid,mushroom,heart,glass,Ions,vehicle,tic,wave,   wdbc,segment,sat
#wpbc,wine,sonar,tic-tac-toe,dermatology,kr-vs-kp,breast-cancer-wisconsin,letter-recognition
#mushroom segmentation shuttle Ticdata2000
#enron 0.87368, 0.12632
#medical 0.99591, 0.00409
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import *
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.datasets import load_digits
import os

def experiments(data_path):
	file_list=os.listdir('./test_data/')
	tables=[]
	for i in file_list:
		print(i)
		data=read_data(data_path+i,',')
		rows=np.size(data,axis=0); cols=np.size(data,axis=1)
		cond=data[:,:-1]
		deci=data[:,-1:].reshape(-1)
		cond_partition=get_partition(cond)
		pos,bnd=pos_region(cond_partition,deci)
		pnum=len(pos); bnum=len(bnd); cols=np.size(data,axis=1)
		# print("The size of universe |U|=%d, the number of attributes |C|=%d, and percentage of POS and BND is %d: %0.4f, %d: %0.4f" % (pnum+bnum,cols, pnum, pnum/(pnum+bnum), bnum, bnum/(pnum+bnum)))
		reduct=reducer(cond,deci)
		types={'POS','GER','DIS','MDS','IND'}
		for rtype in types:
			# reduct.granularity_search(rtype)
			t0 = timeit.default_timer()
			res1=reduct.granularity_approximation(rtype)
			t1 = timeit.default_timer()
			res2=reduct.QGARA_FS(rtype)
			t2 = timeit.default_timer()
			try:
				acc1=test_accuracy(cond,deci)
				acc2=test_accuracy(cond[:,res1],deci)
				acc3=test_accuracy(cond[:,res2],deci)
				tables.append([i,rows,cols,rtype,"granularity_approximation", res1, t1-t0, "QGARA_FS", res2, t2-t1, acc1, acc2, acc3])
			except:
				print("error happen: ",rtype)
				print('GRANU: ',res1,'\nQGARA: ',res2)
				continue
	df = pd.DataFrame(tables)
	df.to_csv(data_path+'experiments.csv', sep=',', header=False, index=False)

@res
def test_accuracy(X,Y):
	# x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
	skf = StratifiedKFold(n_splits=10)
	rf_acc=0; nb_acc=0; kn_acc=0; dt_acc=0;
	for train_index, test_index in skf.split(X, Y):
		x_train, x_test = X[train_index], X[test_index]
		y_train, y_test = Y[train_index], Y[test_index]
		rf = RandomForestClassifier(n_estimators=45, class_weight="balanced")
		rf.fit(x_train, y_train)
		rf_acc=rf_acc+accuracy_score(y_test,rf.predict(x_test))
		bnb = MultinomialNB()
		bnb.fit(x_train, y_train)
		nb_acc=nb_acc+accuracy_score(y_test,bnb.predict(x_test))
		kn=KNeighborsClassifier(9)
		kn.fit(x_train,y_train)
		kn_acc=kn_acc+accuracy_score(y_test,kn.predict(x_test))
		dtre=DecisionTreeClassifier()
		dtre.fit(x_train,y_train)
		dt_acc=dt_acc+accuracy_score(y_test,dtre.predict(x_test))
	return [rf_acc/10,nb_acc/10,kn_acc/10, dt_acc/10]

def timeDifference(cond,deci):
	#show time-consumed change with increasing of data scale 
	attributes_num=np.size(cond,axis=1);rtype='POS'
	if attributes_num<10: return np.zeros((4,9),dtype=float)
	res=[]
	for i in range(1,10):
		tmp=int(attributes_num*i/10)
		red=reducer(cond[:,:tmp],deci[:])
		t0 = timeit.default_timer()
		res1=red.granularity_approximation(rtype)
		t1 = timeit.default_timer()
		red4=red.GS_withoutCoreMax(rtype)
		t2 = timeit.default_timer()
		res2=red.QGARA_FS(rtype)
		t3 = timeit.default_timer()
		red3=red.QGARA_BS(rtype)
		t4 = timeit.default_timer()
		res.append([t3-t2,t4-t3,t1-t0,t2-t1])
		print('\n第%d轮数据：\t'%i, [t3-t2,t4-t3,t1-t0,t2-t1])
		
		# res2=red.QGARA_FS(rtype)
		# red.GS_withoutCore(rtype)
		# red.GS_withoutMax(rtype)
	p=np.array(res)
	return p.T

def testObjects():
	dataPathList=['shuttle(int).txt','mushroom.txt','ticall.txt','segmentation(int).txt','pima-indians-diabetes(int).txt','splice.txt','dermatology.txt','wdbc(int).txt','CNAE9.txt','semeion.txt','DNA.txt','connect4.txt']
	timedif=np.zeros((60,9),dtype=float); rtype='POS'
	for i in range(len(dataPathList)):
		data=read_data('input/'+dataPathList[i],',')
		cond=data[:,:-1]; deci=data[:,-1].reshape(-1)
		p=timeDifference(cond,deci)
		timedif[i*4:i*4+4,:]=p
	ex=pd.DataFrame(timedif)
	ex.to_csv('tmpResult2.csv',encoding='utf-8')

def entropy(m:np.ndarray)->float:
	#input: a matrix, all elements are assumed 
	#return the entropy
	size_U=np.size(m,axis=0)
	if len(np.shape(m))>1:
		m=self.get_partition(m)
	return -sum([item/size_U*np.log2(item/size_U) for _,item in Counter(m).items()])

def condition_entropy(Y:np.ndarray,X:np.ndarray)->float:
	#input two matrix
	#return H(Y|X)
	if len(np.shape(X))>1:
		partition_X=self.get_partition(X)
	else:
		partition_X=X
	value2index=dict()
	size_U=np.size(X,axis=0)
	for i in range(size_U):
		if value2index.get(partition_X[i])==None:
			value2index[partition_X[i]]=[]
		value2index[partition_X[i]].append(i)
	cond_entropy=0.
	for _,item in value2index.items():
		cond_entropy+=len(item)/size_U*entropy(np.take(Y,item,axis=0)) 
	return cond_entropy


if __name__=='__main__':
	X,y =load_digits(return_X_y=True)
	test_accuracy(X,y)
	num_attrs=X.shape[1]
	X_new=SelectKBest(chi2,k=int(num_attrs*0.4)).fit_transform(X,y)
	test_accuracy(X_new,y)
	exit()
	base_path='C:/Users/Brick/硕士数据/我的研究/1-粒度空间/graduate2-granularity search/program-reduction/input'
	data_path=base_path+'/dermatology.txt'#'./data/Input/TicdataValid.txt'
	rtype='POS'#{'POS','GER','DIS','MDS','IND'}
	data=read_data(data_path,','); rows=np.size(data,axis=0); cols=np.size(data,axis=1)
	cond=data[:,:-1]; deci=data[:,-1].reshape(-1)
	# print("class num",len(set(deci)))
	# cond_partition=get_partition(cond); pos,bnd=pos_region(cond_partition,deci); pnum=len(pos); bnum=len(bnd); print("The size of universe |U|=%d, the number of attributes |C|=%d, and percentage of POS and BND is %d: %0.4f, %d: %0.4f" % (pnum+bnum,cols, pnum, pnum/(pnum+bnum), bnum, bnum/(pnum+bnum)))
	red=reducer(cond,deci)
	# res1=red.granularity_approximation(rtype)
	res9=red.granularity_search(rtype)
	# res2=red.QGARA_FS(rtype)
	# red3=red.QGARA_BS(rtype)
	# print(red3)
	# red4=red.GS_withoutCoreMax1(rtype)
	# red.GS_withoutCore(rtype)
	# red.GS_withoutCoreMax(rtype)
	# red3=red.QGARA_BS(rtype)
	# red.GS_withoutCore(rtype)
	# red.GS_withoutMax(rtype)
	# print([i+1 for i in res1])
	# print([i+1 for i in res2])
	# test_accuracy(cond, deci)
	# test_accuracy(cond[:,res1],deci)
	# test_accuracy(cond[:,res2],deci)