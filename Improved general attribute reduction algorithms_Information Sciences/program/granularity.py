#-*- encoding: UTF-8 -*-
from math import *
import numpy as np
import copy
from collections import  Counter
from sklearn import preprocessing
import re
import time, timeit
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import *
from sklearn import linear_model
from sklearn import svm

def clock(func):
    def clocked(*args):
        t0 = timeit.default_timer()
        result = func(*args)
		# for i in range(0,9):
		# 	func(*args)
        elapsed = timeit.default_timer() - t0
        name = func.__name__
        print('Function:%s()\nResult: %r\n---Time Consuming: %0.8f s' % (name,result,elapsed))
        return result
    return clocked

class Reducer(object):
    def __init__(self,path:str):
        # path: the TXT file path of data
        # assumption: the quantity of decision attributes is 1. 
        with open(path,mode='r',encoding='utf8') as fin:
            tmp=np.array([re.split('[,|\t|\ ]{1,}',i.strip()) for i in fin.readlines()])
        data=np.zeros(tmp.shape,dtype=int)
        self.le = preprocessing.LabelEncoder()
        for c in range(tmp.shape[1]):
            data[:,c]=self.le.fit_transform(tmp[:,c])
        self.label_classes=set(data[:,-1])# an array storing the domain of decision value
        self.X=data[:,:-1]
        self.Y=data[:,-1]
        self.tgp=None
        self.data_analysis(path.split('/')[-1])
    
    @property
    def X_(self):
        return self.X
    
    @X_.setter
    def X_(self,value:np.ndarray):
        self.X=value
    @property
    def Y_(self):
        return self.Y
    @Y_.setter
    def Y_(self):
        return self.Y
    
    def data_analysis(self,data_name)->None:
        def pos_num():
            labelCollector=dict()#collect the label values of equivalence class
            indexCollector=dict()#collect the index of objects of equivalence class
            pos_bag=set()
            X_part=self.get_partition(self.X)
            for i in range(self.X.shape[0]):
                key=X_part[i]
                if labelCollector.get(key)==None:
                    labelCollector[key]=self.Y[i]#collect label
                    indexCollector[key]=[i]#collect the index of object
                    pos_bag.add(key)
                else:
                    if key in pos_bag:
                        if self.Y[i]!=labelCollector[key]:#boundary region object
                            pos_bag.remove(key)
                            indexCollector.pop(key)
                        else:#positive region object
                            indexCollector[key].append(i)
            pos_num=0
            for key in pos_bag:
                pos_num+=len(indexCollector[key])
            return pos_num
        object_num=self.X.shape[0]
        attribute_num=self.X.shape[1]
        decision_classes_num=len(set(self.Y))
        dependency_degree=pos_num()/object_num
        print('data name: ',data_name,'objects_num: ', object_num,'attributes_num: ', attribute_num, 'decision_classes_num: ',decision_classes_num,'dependency_degree: ', dependency_degree,sep=' ')
    
    @clock
    def gen_POS_TGP(self):
        #generate the target granularity of Positive region preservation reduction
        labelCollector=dict()#collect the label values of equivalence class
        indexCollector=dict()#collect the index of objects of equivalence class
        pos_bag=set()
        X_part=self.get_partition(self.X)
        for i in range(self.X.shape[0]):
            key=X_part[i]
            if labelCollector.get(key)==None:
                labelCollector[key]=self.Y[i]#collect label
                indexCollector[key]=[i]#collect the index of object
                pos_bag.add(key)
            else:
                if key in pos_bag:
                    if self.Y[i]!=labelCollector[key]:#boundary region object
                        pos_bag.remove(key)
                        indexCollector.pop(key)
                    else:#positive region object
                        indexCollector[key].append(i)
        tgp=np.full(self.X.shape[0],fill_value=-1)
        for key in pos_bag:
            for x_index in indexCollector[key]:
                tgp[x_index]=self.Y[x_index]
        self.tgp=tgp
    
    @clock
    def gen_POS_TGP_list(self):
        #generate the target granularity of Positive region preservation reduction
        x_value=[]
        d_set=[]
        index_x=[]
        X_part=self.get_partition(self.X)
        for i in range(self.X.shape[0]):
            key=X_part[i]
            if key in x_value:
                index=x_value.index(key)
                d_set[index].add(self.Y_[i])
                index_x[index].append(i)
            else:
                x_value.append(key)
                d_set.append(set([self.Y_[i]]))
                index_x.append([i])
        tgp=np.full(self.X.shape[0],fill_value=-1)
        for i in range(len(d_set)):
            if len(d_set[i])==1:
                for j in index_x[i]:
                    tgp[j]=self.Y_[j]
        self.tgp=tgp

    def gen_GDP_TGP(self):
        #generate the target granularity of generalized decision preservation reduction
        labelCollector=dict()#collect the label values of equivalence class
        indexCollector=dict()#collect the index of objects of equivalence class
        X_part=self.get_partition(self.X)
        for i in range(self.X.shape[0]):
            key=X_part[i]
            if labelCollector.get(key)==None:
                labelCollector[key]={self.Y[i]}#collect label
                indexCollector[key]=[i]#collect the index of object
            else:
                labelCollector[key].add(self.Y[i])
                indexCollector[key].append(i)
        tgp=np.zeros(self.X.shape[0])
        decisionEncoder=dict()#map a decision value set to a integer number 
        for key,item in labelCollector.items():
            generalized_decision=str(item)#if set_A==set_B then str(set_A)==str(set_B)
            if decisionEncoder.get(generalized_decision)==None:#generalized decision encoder
                decisionEncoder[generalized_decision]=len(decisionEncoder)
            for x_index in indexCollector[key]:
                tgp[x_index]=decisionEncoder[generalized_decision]
        self.tgp=tgp

    def gen_DP_TGP(self):
        #generate the target granularity of distribution preservation reduction
        indexCollector=dict()#collect the index of objects of equivalence class
        distributionCollector=dict()#collect the label values distribution of equivalence class
        X_part=self.get_partition(self.X)
        for i in range(self.X.shape[0]):
            key=X_part[i]
            if distributionCollector.get(key)==None:
                distributionCollector[key]={j:0 for j in self.label_classes}#collect label distribution
                distributionCollector[key][self.Y[i]]=distributionCollector[key][self.Y[i]]+1
                indexCollector[key]=[i]#collect the index of object
            else:
                distributionCollector[key][self.Y[i]]=distributionCollector[key][self.Y[i]]+1
                indexCollector[key].append(i)
        tgp=np.zeros(self.X.shape[0])
        distributionEncoder=dict()#map a decision value set to a string object 
        for key,item in distributionCollector.items():
            size_ec=len(indexCollector[key])#the size of equivalence class
            for k in self.label_classes:
                item[k]=item[k]/size_ec
            decision_distribution=str(item)
            if distributionEncoder.get(decision_distribution)==None:#generalized decision encoder
                distributionEncoder[decision_distribution]=len(distributionEncoder)
            for x_index in indexCollector[key]:
                tgp[x_index]=distributionEncoder[decision_distribution]
        self.tgp=tgp
    
    def gen_MDP_TGP(self):
        #generate the target granularity of maximum distribution preservation reduction
        indexCollector=dict()#collect the index of objects of equivalence class
        distributionCollector=dict()#collect the label values distribution of equivalence class
        X_part=self.get_partition(self.X)
        for i in range(self.X.shape[0]):
            key=X_part[i]
            if distributionCollector.get(key)==None:
                distributionCollector[key]={j:0 for j in self.label_classes}#collect label distribution
                distributionCollector[key][self.Y[i]]=1
                indexCollector[key]=[i]#collect the index of object
            else:
                distributionCollector[key][self.Y[i]]+=1
                indexCollector[key].append(i)
        tgp=np.zeros(self.X.shape[0])
        distributionEncoder=dict()#map a decision value set to a string object 
        for key,item in distributionCollector.items():
            max_decision=set()
            max_freq=0
            for k in self.label_classes:#compute decision values with maximum distribution
                if item[k]>max_freq:
                    max_freq=item[k]
                    max_decision={k}
                elif item[k]==max_freq:
                    max_decision.add(k)
            max_distribution=str(max_decision)
            if distributionEncoder.get(max_distribution)==None:#generalized decision encoder
                distributionEncoder[max_distribution]=len(distributionEncoder)
            for x_index in indexCollector[key]:
                tgp[x_index]=distributionEncoder[max_distribution]
        self.tgp=tgp

    def gen_RIR_TGP(self):
        #generate the target granularity of relative indiscernibility relation preservation reduction
        labelCollector=dict()#collect the label values of equivalence class
        indexCollector=dict()#collect the index of objects of equivalence class
        posCollector=set()
        X_part=self.get_partition(self.X)
        for i in range(self.X.shape[0]):
            key=X_part[i]
            if labelCollector.get(key)==None:
                labelCollector[key]=set([self.Y[i]])
                indexCollector[key]=[i]#collect the index of object
                posCollector.add(key)
            else:
                labelCollector[key].add(self.Y[i])
                indexCollector[key].append(i)
                if key in posCollector and len(labelCollector[key])>1:
                    posCollector.remove(key)
        tgp=X_part
        for key in posCollector:
            for obj in indexCollector[key]:
                tgp[obj]=-self.Y[obj]
        self.tgp=tgp
    
    def gen_TGP(self,reduction_type:str):
        if reduction_type=='POS':
            self.gen_POS_TGP()
            # self.gen_POS_TGP_list()
        elif reduction_type=='GER':
            self.gen_GDP_TGP()
        elif reduction_type=='DIS':
            self.gen_DP_TGP()
        elif reduction_type=='MDS':
            self.gen_MDP_TGP()
        elif reduction_type=='IND':
            self.gen_RIR_TGP()
        else:
            raise ValueError('Error happened in Reducer.gen_TGP(reduction_type)...')
    
    def get_cir(self,data:np.ndarray):
        # data: a 1*n array
        # description: get the cardinality of the indiscernibility relation of universe with respect to a attribute
        cardinality=0
        for _,freq in Counter(data).items():
            cardinality+=freq**2
        return cardinality
    
    def get_partition(self,data:np.ndarray):
        #take two dimensional array as input
        #return array with same distribution
        #for example
        #in_array:	output:
        #	s 1 s 	1
        #	1 2 1 	2
        #	1 2 1 	2
        #	2 2 1 	3
        #output domain value [1,+oo]
        if len(data.shape)==1:
            data.resize(data.shape[0],1)
        rows=data.shape[0]
        cols=data.shape[1]
        partition=np.array([0 for i in range(rows)]); value2dstr=dict(); valnum=1;
        for i in range(rows):#input list(cols) of value return equivalence class id
            tmp=value2dstr
            for j in range(0,cols):
                if tmp.get(data[i][j])==None:#new equaivalence class
                    partition[i]=valnum;#return a num
                    for k in range(j,cols-1):
                        tmp[data[i][k]]=dict()
                        tmp=tmp[data[i][k]]
                    try:
                        tmp[data[i][cols-1]]=valnum
                    except IndexError:
                        print('Error happened in Reducer.get_partition()',type(data[i]),data[i])
                    valnum+=1
                    break
                else:
                    if j+1==cols:#last node
                        partition[i]=tmp[data[i][j]]
                    else:#next node
                        tmp=tmp[data[i][j]]
        return partition

    def get_relative_discerniblity(self,R:np.ndarray,D:np.ndarray):
        #compute W_\delta(R|D). more detail can be found in 'quick general reduction algorithms for inconsistent decision table'
        if len(R.shape)==1: R=R.reshape(R.shape[0],1)
        if len(D.shape)==1: D=D.reshape(D.shape[0],1)
        relative_discernibility=self.X.shape[0]**2-self.get_cir(self.get_partition(R)) \
            -self.get_cir(self.get_partition(D))+self.get_cir(self.get_partition(np.hstack((R,D))))
        return relative_discernibility
    
    def sort_universe(self,universe:list,attributes:list):
        #radix sort: Sort(U,C)
        for i in attributes:
            vmin=min(self.X[:,i])
            vmax=max(self.X[:,i])
            bucket=[[] for j in range(vmin,vmax+1)]
            for j in universe:
                bucket[self.X[j,i]-vmin].append(j)
            tmp_universe=[]
            for k in bucket:
                tmp_universe.extend(k)
            universe=tmp_universe
        return universe

    def get_partition_using_sort_result(self,universe:list,attributes:list):
        #get the partition of universe with respect to attributes 
        partition=np.full(len(universe),0)
        index=0
        for i in range(1,self.X.shape[0]):
            obj_pre=universe[i-1]
            obj_foc=universe[i]
            if all(self.X[obj_pre,attributes]==self.X[obj_foc,attributes])==True:
                partition[obj_foc]=index
            else:
                index=index+1
                partition[obj_foc]=index
        return partition
    
    def isIRPR(self,attributes:list,reduction_type:str)->bool:
        # return the judgement of the attributes is or not a granularity element of related granularity space.
        #if True is returned, then the attributes is a reduct or a super set of reduct, otherwise, it is not
        self.gen_TGP(reduction_type)
        universe=[i for i in range(self.X.shape[0])]
        universe_X=self.sort_universe(universe,attributes)
        universe_a=self.sort_universe(universe,attributes)
        part_X=self.get_partition_using_sort_result(universe_X,attributes)
        part_a=self.get_partition_using_sort_result(universe_a,attributes)
        return self.get_relative_discerniblity(part_X,self.tgp)==self.get_relative_discerniblity(part_a,self.tgp)

    def get_core_radix_rd(self,stop_point:int):
        core=list()
        not_red=list()
        attributes=[i for i in range(1,self.X.shape[1])]
        universe=[i for i in range(self.X.shape[0])]
        universe=self.sort_universe(universe,attributes)#Sort(U,C)
        partition=self.get_partition_using_sort_result(universe,attributes)
        if self.get_relative_discerniblity(partition,self.tgp)<stop_point:
            core.append(0)
        else: not_red.append(0)
        for i in range(0,self.X.shape[1]-1):
            attributes[i]=i
            universe=self.sort_universe(universe,[i])
            partition=self.get_partition_using_sort_result(universe,attributes)
            if self.get_relative_discerniblity(partition,self.tgp)<stop_point:
                core.append(i+1)
            else: not_red.append(i+1)
        return core,not_red

    #@clock
    def QGARA_FS(self,reduction_type:str):
        self.gen_TGP(reduction_type)
        stop_point=self.get_relative_discerniblity(self.X,self.tgp)
        reduct,not_red=self.get_core_radix_rd(stop_point)
        universe=[i for i in range(self.X.shape[0])]
        red_score=0
        if len(reduct)>0:
            red_score=self.get_relative_discerniblity(self.X[:,reduct],self.tgp)
            universe=self.sort_universe(universe,reduct)
        while red_score<stop_point:
            max_attr=None; max_point=0;max_universe=None
            for i in not_red:#compute significance of attributes
                tmp_universe=self.sort_universe(universe,[i])
                if len(reduct)==0:
                    part=self.get_partition_using_sort_result(tmp_universe,[i])
                    new_point=self.get_relative_discerniblity(part,self.tgp)
                else:
                    part=self.get_partition_using_sort_result(tmp_universe,reduct+[i])
                    new_point=self.get_relative_discerniblity(part,self.tgp)
                if max_point==0 or max_point<new_point:
                    max_point=new_point
                    max_attr=i
                    max_universe=tmp_universe
            #select the attribute with maximal significance
            red_score=max_point
            reduct.append(max_attr)
            not_red.remove(max_attr)
            universe=max_universe
        return sorted([i for i in reduct])
    
    # @clock
    def QGARA_BS(self,reduction_type:str):
        self.gen_TGP(reduction_type)
        stop_point=self.get_relative_discerniblity(self.X,self.tgp)
        attributes=[i for i in range(0,self.X.shape[1])]
        universe=[i for i in range(self.X.shape[0])]
        reduct=[i for i in range(1,self.X.shape[1])]
        universe=self.sort_universe(universe,attributes)#Sort(U,C)
        red_score=self.get_relative_discerniblity(
            self.get_partition_using_sort_result(universe,reduct),#U/C-{a_1}
            self.tgp
        )#W(C|D_\delta)
        flag=False; last_preserve=self.X.shape[1]-1
        if red_score==stop_point:#can be deleted
            flag=True
        else:
            reduct.append(0)
            flag=False
            last_preserve=0
        for i in range(1,self.X.shape[1]):
            reduct.remove(attributes[i])
            if flag:#a_{i-1} can be deleted
                pass
            else:#a_{i-1} can not be deleted
                if last_preserve!=self.X.shape[1]-1:
                    universe=self.sort_universe(universe,[last_preserve])
            partition=self.get_partition_using_sort_result(universe,reduct)
            red_score=self.get_relative_discerniblity(partition,self.tgp)
            if red_score==stop_point:
                flag=True
            else:
                reduct.append(attributes[i])
                flag=False
                last_preserve=attributes[i]
        return sorted([i for i in reduct])
    
    def positive_region_size(self,B:np.ndarray,D:np.ndarray):
	    #compute |POS_B(D)|, B:one dimensional array, D:one dimensional array
        decision_bag=dict()
        pos_bag=set()
        obj_counter=dict()
        for i in range(B.shape[0]):
            if decision_bag.get(B[i])==None:
                decision_bag[B[i]]=D[i]
                obj_counter[B[i]]=1
                pos_bag.add(B[i])
            elif B[i] in pos_bag:
                if decision_bag[B[i]]!=D[i]:#not positive region object
                    pos_bag.remove(B[i])
                    obj_counter.pop(B[i])
                else:
                    obj_counter[B[i]]+=1
        return sum([obj_counter[i] for i in pos_bag])
    
    def boundary_region(self,B:np.ndarray,D:np.ndarray):
	    #compute POS_B(D), B:one dimensional array, D:one dimensional array
        decision_bag=dict()
        bnd_bag=set()
        obj_bag=dict()
        for i in range(B.shape[0]):
            if decision_bag.get(B[i])==None:
                decision_bag[B[i]]=D[i]
                obj_bag[B[i]]=[i]
            else:
                if decision_bag[B[i]]!=D[i]:#not positive region object
                    bnd_bag.add(B[i])
                obj_bag[B[i]].append(i)
        bnd_region=[]
        for i in bnd_bag:
            bnd_region+=obj_bag[i]
        return bnd_region

    def get_core_radix_pz(self,stop_point:int):
        core=list()
        not_red=list()
        attributes=[i for i in range(1,self.X.shape[1])]
        universe=[i for i in range(self.X.shape[0])]
        universe=self.sort_universe(universe,attributes)#Sort(U,C)
        partition=self.get_partition_using_sort_result(universe,attributes)
        if self.positive_region_size(partition,self.tgp)<stop_point:
            core.append(0)
        else: not_red.append(0)
        for i in range(0,self.X.shape[1]-1):
            attributes[i]=i
            universe=self.sort_universe(universe,[i])
            partition=self.get_partition_using_sort_result(universe,attributes)
            if self.positive_region_size(partition,self.tgp)<stop_point:
                core.append(i+1)
            else: not_red.append(i+1)
        return core,not_red
    
    def isGS(self,attributes:list,reduction_type:str)->bool:
        # return the judgement of the attributes is or not an indiscernibility relation preservation reduct.
        #if True is returned, then the attributes is a reduct or a super set of reduct, otherwise, it is not
        self.gen_TGP(reduction_type)
        part_X=self.get_partition(self.X)
        part_a=self.get_partition(self.X[:,attributes])
        return self.positive_region_size(part_X,self.tgp)==self.positive_region_size(part_a,self.tgp)
    
    def entropy(self,m:np.ndarray)->float:
        #input: a matrix, all elements are assumed 
        #return the entropy
        size_U=np.size(m,axis=0)
        if len(np.shape(m))>1:
            m=self.get_partition(m)
        return -sum([item/size_U*np.log2(item/size_U) for _,item in Counter(m).items()])

    def condition_entropy(self,Y:np.ndarray,X:np.ndarray)->float:
        #input two matrix
        #return H(Y|X)
        if len(np.shape(Y))>1:
            partition_Y=self.get_partition(Y)
        else:
            partition_Y=Y
        value2index=dict()
        size_U=np.size(Y,axis=0)
        for i in range(size_U):
            if value2index.get(partition_Y[i])==None:
                value2index[partition_Y[i]]=[]
            value2index[partition_Y[i]].append(i)
        cond_entropy=0.
        for _,item in value2index.items():
            cond_entropy+=len(item)/size_U*self.entropy(np.take(X,item,axis=0)) 
        return cond_entropy
    
    @clock
    def GS(self,reduction_type:str):
        #granularity search
        self.gen_TGP(reduction_type)
        X_tmp=np.copy(self.X)
        tgp_tmp=np.copy(self.tgp)
        reduct,not_red=self.get_core_radix_pz(self.X.shape[0])
        red_score=0; red_part=None
        if len(reduct)>0:
            red_part=self.get_partition(X_tmp[:,reduct])
            bnd_region=self.boundary_region(red_part,tgp_tmp)
            red_score=X_tmp.shape[0]-len(bnd_region)
            X_tmp=np.take(X_tmp,bnd_region,axis=0)#np.compress(max_index,X_tmp,axis=0)
            tgp_tmp=np.take(tgp_tmp,bnd_region,axis=0)#np.compress(max_index,tgp_tmp,axis=0)
            red_part=np.take(red_part,bnd_region,axis=0)
        while red_score<self.X.shape[0]:
            max_part=np.zeros(X_tmp.shape[0])
            select_attribute=None
            max_score=0
            for i in not_red:#compute significance of attributes
                if len(reduct)==0:
                    now_part=X_tmp[:,i]
                else:
                    now_part=self.get_partition(np.vstack((red_part,X_tmp[:,i])).T)
                pos_size=self.positive_region_size(now_part,tgp_tmp)
                if max_score==0 or max_score<pos_size:#guarantee that each iteration we add an attribute into red
                    max_score=pos_size
                    max_part=now_part
                    select_attribute=i
            reduct.append(select_attribute)
            not_red.remove(select_attribute)
            bnd_region=self.boundary_region(max_part,tgp_tmp)
            X_tmp=np.take(X_tmp,bnd_region,axis=0)#np.compress(max_index,X_tmp,axis=0)
            tgp_tmp=np.take(tgp_tmp,bnd_region,axis=0)#np.compress(max_index,tgp_tmp,axis=0)
            red_part=np.take(max_part,bnd_region,axis=0)#np.compress(max_index,red_part,axis=0)
            red_score+=max_score
            # print('reduct: '+str(len(reduct))+',\t attribute: '+str(select_attribute)+',\t now_point:  '+str(max_score)+',\t boundary region: '+str(len(bnd_region)))
        return sorted([i for i in reduct])
    
    # @clock
    def GSV(self,reduction_type:str):
        #A variant of granularity search algorithm
        self.gen_TGP(reduction_type)
        X_tmp=np.copy(self.X)
        tgp_tmp=np.copy(self.tgp)
        red_score=0; red_part=np.zeros(self.X.shape[0])
        not_red=[i for i in range(self.X.shape[1])]
        reduct=[]
        tmp_red=copy.deepcopy(not_red)
        while red_score<self.X.shape[0]:
            add_attributes=False
            for i in not_red:#compute significance of attributes
                if len(reduct)==0:
                    now_part=X_tmp[:,i]
                else:
                    now_part=self.get_partition(np.vstack((red_part,X_tmp[:,i])).T)
                pos_size=self.positive_region_size(now_part,tgp_tmp)
                # print('reduct: ',reduct,'attributes:',i,'pos_size:',pos_size)
                if pos_size>0:
                    add_attributes=True
                    reduct.append(i)
                    tmp_red.remove(i)
                    bnd_region=self.boundary_region(now_part,tgp_tmp)
                    X_tmp=np.take(X_tmp,bnd_region,axis=0)#np.compress(max_index,X_tmp,axis=0)
                    tgp_tmp=np.take(tgp_tmp,bnd_region,axis=0)#np.compress(max_index,tgp_tmp,axis=0)
                    red_part=np.take(now_part,bnd_region,axis=0)#np.compress(max_index,red_part,axis=0)
                    red_score+=pos_size
            if add_attributes==False and red_score<self.X.shape[0]:
                #if all the attributes get zero significance, then put an arbitrary attribute into reduct
                arbitrary_attribute=tmp_red[0]
                reduct.append(arbitrary_attribute)
                tmp_red.remove(arbitrary_attribute)#for the sake that arbitrary attribute getting zero significance, so updating X_tmp,tgp_tmp,red_score is unnecessary
                not_red=tmp_red
                red_part=self.get_partition(np.vstack((red_part,X_tmp[:,arbitrary_attribute])).T)
        return sorted([i for i in reduct])
    @clock
    def Q_MDRA(self):
        def gamma_md(part:list)->float:
            gamma=0.
            value2index=dict()
            size_U=self.X.shape[0]
            for i in range(size_U):
                if value2index.get(part[i])==None:
                    value2index[part[i]]=[]
                value2index[part[i]].append(i)
            for _,item in value2index.items():
                gamma+=max([item for _,item in Counter(np.take(self.Y,item,axis=0)).items()])
            return gamma
        reduct,not_red=[],[i for i in range(self.X.shape[1])]
        red_score=0; red_part=None
        C_part=self.get_partition(self.X)
        max_part=np.zeros(self.X.shape[0])
        red_part=np.zeros(self.X.shape[0])
        stop_point=gamma_md(C_part)
        while red_score<stop_point:
            #select the best attribute in $not_red$(a variable)
            select_attribute=None
            max_score=0
            for i in not_red:
                if len(reduct)==0:
                    now_part=self.X[:,i]
                else:
                    now_part=self.get_partition(np.vstack((red_part,self.X[:,i])).T)
                gamma=gamma_md(now_part)#self.positive_region_size(now_part,tgp_tmp)
                if max_score==0 or max_score<gamma:#guarantee that each iteration we add an attribute into red
                    max_score=gamma
                    max_part=now_part
                    select_attribute=i
            reduct.append(select_attribute)
            not_red.remove(select_attribute)
            red_part=max_part
            red_score=max_score
        return sorted([i for i in reduct])
    @clock
    def GS_entropy(self,reduction_type:str):
            #granularity search
            self.gen_TGP(reduction_type)
            X_tmp=np.copy(self.X)
            tgp_tmp=np.copy(self.tgp)
            reduct,not_red=[],[i for i in range(self.X.shape[1])]#self.get_core_radix_pz(self.X.shape[0])
            red_part=np.zeros(X_tmp.shape[0]);max_part=np.zeros(X_tmp.shape[0])
            # if len(reduct)>0:
            #     red_part=self.get_partition(X_tmp[:,reduct])
            #     bnd_region=self.boundary_region(red_part,tgp_tmp)
            #     red_score=X_tmp.shape[0]-len(bnd_region)
            #     X_tmp=np.take(X_tmp,bnd_region,axis=0)#np.compress(max_index,X_tmp,axis=0)
            #     tgp_tmp=np.take(tgp_tmp,bnd_region,axis=0)#np.compress(max_index,tgp_tmp,axis=0)
            #     red_part=np.take(red_part,bnd_region,axis=0)
            while X_tmp.shape[0]!=0:
                select_attribute=None
                max_score=self.X.shape[0]
                for i in not_red:#compute significance of attributes
                    if len(reduct)==0:
                        now_part=X_tmp[:,i]
                    else:
                        now_part=self.get_partition(np.vstack((red_part,X_tmp[:,i])).T)
                    cond_entropy=self.condition_entropy(tgp_tmp,now_part)#self.positive_region_size(now_part,tgp_tmp)
                    if max_score>cond_entropy:#guarantee that each iteration we add an attribute into red
                        max_score=cond_entropy
                        max_part=now_part
                        select_attribute=i
                reduct.append(select_attribute)
                not_red.remove(select_attribute)
                bnd_region=self.boundary_region(max_part,tgp_tmp)
                X_tmp=np.take(X_tmp,bnd_region,axis=0)#np.compress(max_index,X_tmp,axis=0)
                tgp_tmp=np.take(tgp_tmp,bnd_region,axis=0)#np.compress(max_index,tgp_tmp,axis=0)
                red_part=np.take(max_part,bnd_region,axis=0)#np.compress(max_index,red_part,axis=0)
                red_score=max_score
                # print('reduct: '+str(len(reduct))+',\t attribute: '+str(select_attribute)+',\t now_point:  '+str(max_score)+',\t boundary region: '+str(len(bnd_region)))
            return sorted([i for i in reduct])

def timeComparisonForDifferentData():
    # time consumptions of reduction algorithms in different data sets
    base_path='C:/Users/Brick/硕士数据/我的研究/1-粒度空间/graduate2-granularity search/program-reduction/input/'
    file_names=['shuttle(int).txt','mushroom.txt','ticall.txt','segmentation(int).txt','pima-indians-diabetes(int).txt','splice.txt','dermatology.txt','wdbc(int).txt','CNAE9.txt','semeion.txt','DNA.txt','connect4.txt']#
    reduction_types=['POS','GER','DIS','MDS','IND']
    time_comparison=[]
    res_comparison=[]
    for fn in file_names:
        red=Reducer(base_path+fn)
        time_row=[]
        res_row=[]
        for rt in reduction_types:
            t0 = timeit.default_timer()
            res_fs=red.QGARA_FS(rt)
            t1 = timeit.default_timer()
            res_bs=red.QGARA_BS(rt)
            t2 = timeit.default_timer()
            res_gs=red.GS(rt)
            t3 = timeit.default_timer()
            res_gsv=red.GSV(rt)
            t4 = timeit.default_timer()
            time_row.extend([t1-t0,t2-t1,t3-t2,t4-t3])
            res_row.extend([res_fs,res_bs,res_gs,res_gsv])
            print('数据集：%s\t 约简类型：%s\n\t 时间耗费——QGARA-FS: %f,  QGARA-BS: %f,  GS: %f,  GSV: %f'%(fn,rt,t1-t0,t2-t1,t3-t2,t4-t3))
            if red.isGS(res_fs,rt)==False or red.isGS(res_bs,rt)==False or red.isIRPR(res_gs,rt)==False or red.isIRPR(res_gsv,rt)==False:
                print('Error discovered in 数据集：%s\t 约简类型：%s'%(fn,rt))
        if len(reduction_types)==1:
            time_row*=5
            res_row*=5
        time_comparison.append(time_row)
        res_comparison.append(res_row)
        num=len(time_comparison)
        timecmp=pd.DataFrame(time_comparison,index=file_names[:num],columns=['POS']*4+['GER']*4+['DIS']*4+['MDS']*4+['IND']*4)
        timecmp.to_csv(base_path+'time_comparison.csv',encoding='utf8')
        rescmp=pd.DataFrame(res_comparison,index=file_names[:num],columns=['POS']*4+['GER']*4+['DIS']*4+['MDS']*4+['IND']*4)
        rescmp.to_csv(base_path+'reduct_comparison.csv',encoding='utf8')
        if fn=='wdbc(int).txt': reduction_types=['POS']

def str2list(para:str):
    return list(set(json.loads(para)))

#logisticRegression, svm, decisionTree
def get_accuracy(X,Y):
    # x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    skf = StratifiedKFold(n_splits=10)
    rf_acc=[]; nb_acc=[]; kn_acc=[]; dt_acc=[]; svm_acc=0.
    for train_index, test_index in skf.split(X, Y):
        x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        # rf=RandomForestClassifier(n_estimators=40)
        # rf.fit(x_train, y_train)
        # rf_acc=rf_acc+accuracy_score(y_test,rf.predict(x_test))
        bnb = MultinomialNB()
        bnb.fit(x_train, y_train)
        nb_acc.append(accuracy_score(y_test,bnb.predict(x_test)))
        # kn=KNeighborsClassifier(9)
        # kn.fit(x_train,y_train)
        # kn_acc=kn_acc+accuracy_score(y_test,kn.predict(x_test))
        dtre=DecisionTreeClassifier()
        dtre.fit(x_train,y_train)
        dt_acc.append(accuracy_score(y_test,dtre.predict(x_test)))
        # sv=svm.SVC(gamma='scale')
        # sv.fit(x_train,y_train)
        # svm_acc=svm_acc+accuracy_score(y_test,sv.predict(x_test))
    return nb_acc,dt_acc#[rf_acc/10,nb_acc/10,kn_acc/10, dt_acc/10,svm_acc/10]

from sklearn.feature_selection import SelectKBest, chi2
def reduct_analysis():
    base_path='C:/Users/Brick/硕士数据/我的研究/1-粒度空间/graduate2-granularity search/program-reduction/input/'
    file_names=['shuttle(int).txt','mushroom.txt','ticall.txt','segmentation(int).txt','pima-indians-diabetes(int).txt','splice.txt','dermatology.txt','wdbc(int).txt','CNAE9.txt','semeion.txt','DNA.txt','connect4.txt']#
    reducts=pd.read_csv(base_path+'reduct_comparison.csv')
    col_list=[2,3,4,5,18,19,20,21]
    res=[]; accuracy_list=[]
    col_name=[]
    for i in [1]:#[2,3,4,6,7,8,9,10,11,12]:#range(1,2):
        red=Reducer(base_path+file_names[i])
        tmp_accuracy=[]#5
        tmp_accuracy.extend(get_accuracy(red.X_,red.Y_))
        if i==1: col_name.extend(['naiveBayes_O','decisionTree_O'])
        for j in col_list:
            reduct=str2list(reducts.iloc[i,j])
            tmp_accuracy.extend(get_accuracy(np.take(red.X_,reduct,axis=1),red.Y_))#8*5=40
            if i==1: col_name.extend(['naiveBayes','decisionTree'])
        res.append(diff(tmp_accuracy))
        if file_names[i]=='connect4.txt':
            break
        # for j in [4,20]:
        #     reduct=str2list(reducts.iloc[i,j])
        #     tmp_accuracy.extend(get_accuracy(SelectKBest(chi2,k=len(reduct)).fit_transform(red.X_,red.Y_),red.Y_))#8*5
        #     if i==1: col_name.extend(['naiveBayes_ST','decisionTree_ST'])
        # accuracy_list.append(tmp_accuracy)#11*45
        # acc=pd.DataFrame(accuracy_list,index=reducts.iloc[1:i+1,1],columns=col_name)#,index=file_names[11],columns=['randomForest','naiveBayes','KNN','decisionTree','svm']*11)
        # acc.to_excel(base_path+'accuracy_comparison_Mushroom.xlsx',encoding='utf8')
        # print(reducts.iloc[i,1],' have finished!')
    # np.save('./testResult.npy',res)
    print(res)

import scipy
def isDiff(a,b):
    #双样本正态同方差：原假设H_0:\mu_0=\mu_1,备选假设H_1:\mu_0!=\mu_1
    sta,p_val=scipy.stats.ttest_ind(a,b,0)
    if p_val>0.05:
        return False
    else:
        return True

def diff(tmp):
    assert len(tmp)==18
    res=[]
    for i in [2,3,4,5,10,11,12,13]:
        res.append(isDiff(tmp[i],tmp[i+4]))
    return res

def timeComparisonWithSingleData():
    base_path='C:/Users/Brick/硕士数据/我的研究/1-粒度空间/graduate2-granularity search/program-reduction/input/'
    file_names=['connect4.txt','mushroom.txt','ticall.txt','segmentation(int).txt','splice.txt','dermatology.txt','wdbc(int).txt','CNAE9.txt','semeion.txt','DNA.txt','connect4.txt']#
    red_type=['POS']#,'IND']# POS GER DIS MDS IND
    timeCmp=[]
    for fn in file_names:
        print(fn)
        red=Reducer(base_path+fn)
        data=copy.deepcopy(red.X_)
        qfs_t=[]
        qbs_t=[]
        gs_t=[]
        gsv_t=[]
        for rt in red_type:
            print('\t'+rt)
            print('\t\t'+'objects')
            #objects
            for i in range(1,10):
                objects=int(data.shape[0]*i/10)
                red.X_=np.take(data,[i for i in range(objects)],axis=0)
                t0 = timeit.default_timer()
                res_fs=red.QGARA_FS(rt)
                t1 = timeit.default_timer()
                res_bs=red.QGARA_BS(rt)
                t2 = timeit.default_timer()
                res_gs=red.GS(rt)
                t3 = timeit.default_timer()
                res_gsv=red.GSV(rt)
                t4 = timeit.default_timer()
                
                qfs_t.append(t1-t0)
                qbs_t.append(t2-t1)
                gs_t.append(t3-t2)
                gsv_t.append(t4-t3)
        
        for rt in red_type:
            print('\t'+rt)
            print('\t\t'+'attributes')
            #attributes
            for i in range(1,10):
                attributes=int(data.shape[1]*i/10)
                red.X_=np.take(data,[i for i in range(attributes)],axis=1)
                t0 = timeit.default_timer()
                res_fs=red.QGARA_FS(rt)
                t1 = timeit.default_timer()
                res_bs=red.QGARA_BS(rt)
                t2 = timeit.default_timer()
                res_gs=red.GS(rt)
                t3 = timeit.default_timer()
                res_gsv=red.GSV(rt)
                t4 = timeit.default_timer()

                qfs_t.append(t1-t0)
                qbs_t.append(t2-t1)
                gs_t.append(t3-t2)
                gsv_t.append(t4-t3)
        timeCmp.extend([qfs_t,qbs_t,gs_t,gsv_t])
        singleData=pd.DataFrame(timeCmp)
        singleData.to_csv(base_path+'singleData.csv',encoding='utf8')

def prepare_data_for_IJCRS():
    #just preprocess arff file to the way I wanted
    from preprocess import arff2txt
    data_path='C:/Users/Brick/硕士数据/我的研究/1-粒度空间/graduate2-granularity search/program-reduction/data/CAIM/'
    arff2txt(data_path)
    #wpdc, wine, sat, segment, wdbc, waveform, vehicle, Ions, Glass, Heart, Sonar, Pid

if __name__=='__main__':
    # timeComparisonForDifferentData()
    # exit()
    reduct_analysis()
    exit()
    # timeComparisonWithSingleData()
    # exit()
    base_path='C:/Users/Brick/硕士数据/我的研究/1-粒度空间/graduate2-granularity search/program-reduction/input/'#CAIM
    file_names_information_sciences=['shuttle(int).txt','mushroom.txt','ticall.txt','segmentation(int).txt','pima-indians-diabetes(int).txt','splice.txt','dermatology.txt','wdbc(int).txt','CNAE9.txt','semeion.txt','DNA.txt','connect4.txt']#
    file_names_IJCRS_2020=['Dis_wpbc(F_CAIM2).txt','Dis_wine(F_CAIM2).txt','Dis_sat(F_CAIM2).txt','Dis_segment(F_CAIM2).txt',\
        'Dis_segment(F_CAIM21).txt','Dis_wdbc(F_CAIM2).txt','Dis_waveform(F_CAIM2).txt','Dis_vehicle(F_CAIM2).txt',\
            'Dis_ino(F_CAIM2).txt','Dis_glass(F_CAIM2).txt','Dis_heart(F_CAIM2).txt','Dis_Sonar(F_CAIM2).txt','Dis_pid(F_CAIM2).txt']
    #wpdc, wine, sat, segment, wdbc, waveform, vehicle, Ions, Glass, Heart, Sonar, Pid
    file_name='connect4.txt'#semeion, CNAE9, DNA, connect4
    red_type='POS'# POS GER DIS MDS IND
    red=Reducer(base_path+file_name)
    res_gsv=red.GSV(red_type)
    exit()
    res_gentropy=red.GS_entropy(red_type)
    res_gs=red.GS(red_type)
    # res_gsv=red.GSV(red_type)
    res_qmdra=red.Q_MDRA()
    res_fs=red.QGARA_FS(red_type)
    # res_bs=red.QGARA_BS(red_type)
    print(red.isGS(res_gentropy,red_type))
    # print(red.isGS(res_fs,red_type),\
    #     red.isGS(res_bs,red_type),\
    #     red.isIRPR(res_gs,red_type),\
    #     red.isIRPR(res_gsv,red_type)\
    # )
    