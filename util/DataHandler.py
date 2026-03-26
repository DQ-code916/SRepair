import numpy as np
import pandas as pd
from util.Assist import *
from time import time
from itertools import combinations
import math
import json
import os
from concurrent.futures import ProcessPoolExecutor
import random




class DataHandler():
    def __init__(self,db,k_T,k_L,ConsAttr=[]):
        if len(ConsAttr)==0:
            self.db=db         
        else: 
            self.db=db[:,ConsAttr]
        self.n, self.m=self.db.shape 
        self.DOM={}      
        self.DomDist={}  
        self.DomDist_dict={} 
        self.TupleDist=[] 
        self.val_id={}     
        self.TList=[]    
        self.k_T=k_T         
        self.k_L=k_L      
        self.DModel={}     
        self.Lijc={}     
        self.Lij={}
        self.Li={}
        self.L=0
        self.Error={}   
        self.LList={}   
        self.ToTopK={}  
        self.Knn={}     
        self.ConsAttr=set()     
        self.db_val_id=np.zeros(db.shape,dtype=int)      



    def DomGenerator(self):        
        for c in range(self.m):
            self.DOM[c]=list(set(self.db[:,c]))


    def CalcDomDist(self,if_CheckMem=False,if_process=False):         
        self.Col_DM={}    
        self.Col_Type={'str':[],'num':[]}
        t1=time()
        for c in range(self.m):
            if if_process:
                t2=time()
                print('CalcDomDist-',c,'Time Cost:',round(t2-t1,3))
                t1=time()
            D=len(self.DOM[c])
            if D==1:
                continue
            if isinstance(self.DOM[c][0],str):
                self.Col_Type['str'].append(c)
            else:
                self.Col_Type['num'].append(c)
                tem=set()
                for i in range(D):
                    for j in range(i+1,D):
                        tem1,tem2=self.DOM[c][i],self.DOM[c][j]
                        d=Num_Dist_Between_2_Ele(self.DOM[c][i],self.DOM[c][j])
                        tem.add(d)
                self.Col_DM[c]=max(tem)
    
    
    def Calc_X(self,l1,l2):    
        X = [
            Str_Dist_Between_2_Ele_Sta(l1[c], l2[c]) if c in self.Col_Type['str'] else 
            Num_Dist_Between_2_Ele_Sta(l1[c], l2[c], self.Col_DM[c]) if c in self.Col_DM else 
            0
            for c in range(len(l1))
        ]
        return X

    def CalcTpDist_m(self,Ig,K,if_CheckMem=False,if_process=False):   
        self.TList={}
        self.Knn={}
        self.TupleDist={i:{} for i in range(self.n)}
        t1=time()
        for i in range(self.n):
            if if_process and i%500==0:
                t2=time()
                print('CalcTpDist-',i,'Time Cost:',t2-t1)
                t1=time()
            for j in range(self.n):
                if i==j:
                    continue
                X=self.Calc_X(self.db[i,:],self.db[j,:])
                self.TupleDist[i][j]=sum(X)
            self.Update_Dist(i,Ig,K)
            self.TupleDist.pop(i)


    def CalcTpDist(self, Ig, K, if_CheckMem=False, if_process=False, seed=42, max_workers=None, sampling_threshold=1e5):
        self.TList = {}
        self.Knn = {}
        self.TupleDist = {i: {} for i in range(self.n)}
        t1 = time()


        if self.n <= sampling_threshold:
            temp = range(self.n)
        else:
            random.seed(seed)
            temp = random.sample(range(self.n), self.n // 300)
        num_workers = max_workers or os.cpu_count()
        chunk_size = self.n // num_workers
        remainder = self.n % num_workers
        chunks = []
        start = 0

        for i in range(num_workers):
            end = start + chunk_size + (1 if i < remainder else 0)
            chunks.append(list(range(start, end)))
            start = end

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for chunk in chunks:
                futures.append(executor.submit(process_chunk, chunk, temp, self.db, self.Calc_X, Ig, K, self.k_T,if_process))
                

            for future in futures:
                chunk_result = future.result()
                for i, Knn_i, TList_i in chunk_result:
                    self.Knn[i] = Knn_i
                    self.TList[i] = TList_i

    def Update_Dist(self,i,Ig,K):      
        key_set=sorted(self.TupleDist[i].keys(), key=lambda x:self.TupleDist[i][x])
        self.Knn[i]=[x for x in key_set[:K]]  
        num_TList=0
        self.TList[i]=[]  
        for j in key_set:             
            if j==i:
                continue 
            if j in Ig:
                self.TList[i].append(j)
                num_TList+=1
                if num_TList>=self.k_T:
                    break 
        self.TupleDist[i]={j:self.TupleDist[i][j] for j in self.TupleDist[i].keys() if j in self.Knn[i] or j in self.TList[i]}



    def Save_Knn_TList(self,DataSet):  
        if DataSet not in {'yeast','iris'}:
            data_to_save = [self.Knn, self.TList]
            path=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+'/data/'+DataSet+'/Knn_TList.txt'
            with open(path, 'w') as file:
                json.dump(data_to_save, file)
    
    def Read_Knn_TList(self,DataSet):           
        path=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+'/data/'+DataSet+'/Knn_TList.txt'
        with open(path, 'r') as file:
            loaded_data = json.load(file)
        Knn, TList = loaded_data
        self.Knn={int(i):Knn[i] for i in Knn.keys()}
        self.TList={int(i):TList[i] for i in TList.keys()}
    
    def Knn_Prune(self,K,Ig):        
        self.TList={i:[] for i in range(self.n)}
        self.Knn={i:self.Knn[i] for i in range(self.n)}
        for i in self.Knn.keys():
            self.Knn[i]=[j for j in self.Knn[i] if j in range(self.n)]
            id=0
            for j in self.Knn[i]:
                if j in Ig:
                    self.TList[i].append(j)
                    id+=1
                    if id==self.k_T:
                        break
            if len(self.Knn[i])>K:
                self.Knn[i]=self.Knn[i][:K]


    def FindTList(self,Ig):   
        self.TList={}
        TupleDistClean=self.TupleDist[:,Ig]
        for i in range(self.n):
            if i in Ig:
                tem=find_top_k(TupleDistClean[i,:],self.k_T+1)[1][1:] 
            else:
                tem=find_top_k(TupleDistClean[i,:],self.k_T)[1]
            self.TList[i]=sorted([Ig[j] for j in tem])



    def DistModel(self,if_process=False): 
        t1=time()
        print("self.n=",self.n)
        for i in range(self.n):
            if if_process and i%500==0:
                t2=time()
                print('DistModel-',i,'Time Cost:',round(t2-t1,3))
                t1=time()
            ADict={}
            self.DModel[i]={}
            for c in self.DOM.keys():
                ADict[c]={'X':np.zeros((math.comb(self.k_T,2)+self.k_T, self.m)),'Y':np.zeros((math.comb(self.k_T,2)+self.k_T, 1))}
                self.DModel[i][c]={}
            if i in self.TList:
                tem=min(self.k_T,len(self.TList[i])) 
            else: 
                continue 

            for j_id in range(tem):
                j=self.TList[i][j_id]
                X=[1]+self.Calc_X(self.db[i,:],self.db[j,:])
                for c in self.DOM.keys():
                    ADict[c]['Y'][j_id]=X[c+1]
                    ADict[c]['X'][j_id,:]=np.concatenate((X[:c+1],X[c+2:]))
            

            id=self.k_T
            for j_id in range(tem):
                for j_id2 in range(j_id+1,tem):
                    try:
                        j,j2=self.TList[i][j_id],self.TList[i][j_id2]
                    except:
                        continue
                    X=[1]+self.Calc_X(self.db[j,:],self.db[j2,:])
                    for c in self.DOM.keys():
                        ADict[c]['Y'][id]=X[c+1]
                        ADict[c]['X'][id,:]=np.concatenate((X[:c+1],X[c+2:]))
                    id+=1
            for c in self.DOM.keys():
                coef,sigma2=TrainDModel_byHand(ADict[c]['X'],ADict[c]['Y'],self.m,self.k_T)
                self.DModel[i][c]['Sigma^2']=sigma2+1e-7             
                coef=coef.T[0]
                self.DModel[i][c]['Model']=np.concatenate((coef[:c+1],-1*np.ones((1,)),coef[c+1:]),axis=0)

    def CalcLoss(self,Ic,Ig,if_CheckMem=False,if_process=False):
        t1=time()
        for i in Ic:
            if if_process and i%500==0:
                t2=time()
                print('CalcLoss-',i,'Time Cost:',round(t2-t1,3))
                t1=time()
            self.Lij[i]={}
            self.Lijc={}
            if i not in self.Knn: 
                continue
            for j in self.Knn[i]:
                
                if j==i:
                    continue
                self.Lijc[j],self.Lij[i][j]={},{}
                X=[1]+self.Calc_X(self.db[i,:],self.db[j,:])
                X=np.array(X)
                for c in self.DOM.keys():
                    error=abs(self.DModel[j][c]['Model'] @ X)
                    self.Lijc[j][c]=error
                self.Lij[i][j]=sum(self.Lijc[j].values())
            self.Li[i]=self.Update_Lij(i,Ig)

    def RelationL_byLoss(self,I):      
        for i in I: 
            self.LList[i]={}
            self.LList[i]=sorted(self.Lij[i], key=self.Lij[i].get)      
            self.ToTopK[i]=[]
        for i in I:
            for j in self.LList[i][:self.k_L]:
                self.ToTopK[j].append(i)

    def FindK_for_Lijc(self,K,Rg='All'):         
        if K>self.n:
            K=self.n
        if Rg=='All':
            for i in range(self.n):
                self.Knn[i]=find_top_k(self.TupleDist[i,:],K)[1][1:]

    def get_combinations(self, List):
        combinations_list = list(combinations(List, self.k_T))
        tuples_list = [tuple(combination) for combination in combinations_list]
        return tuples_list

    def Calc_X_Dict(self,if_CheckMem=False):        
        self.X_Dict={}
        for i in range(self.n):
            self.X_Dict[i]={}
            for j in range(self.n):
                self.X_Dict[i][j]=np.array([1]+[self.DomDist[c][self.db_val_id[i,c],self.db_val_id[j,c]] for c in range(self.m)])

    def Update_Lij(self,i,Ig):         
        tem_dict,in_point,=sorted(self.Lij[i], key=lambda j:self.Lij[i][j]),[]
        num=0
        num2=0
        Li=0
        for j in tem_dict:
            in_point.append(j)
            if num2<self.k_L:    
                Li+=self.Lij[i][j]
                num2+=1
            if j in Ig:        
                num+=1
                if num>=self.k_L:
                    break
        self.Lij[i]={j: self.Lij[i][j] for j  in in_point}
        return Li

    def Update_Lij_m(self,Ig):           
        tem_dict,in_point={},{}
        for i in self.Lij.keys():
            tem_dict[i]=sorted(self.Lij[i], key=self.Lij[i].get)
        for i in tem_dict.keys():
            num=0
            in_point[i]=[]
            for j in tem_dict[i]:
                in_point[i].append(j)
                if j in Ig:
                    num+=1
                    if num>=self.k_L:
                        break
        self.Lij={i:{j: self.Lij[i][j] for j in self.Lij[i].keys() if j in in_point[i]} for i in self.Lij.keys()}



    def CalcLoss_m(self,Ic,Ig,SelectedAttr,if_CheckMem=False,if_process=False):    
        t1=time()
        for i in Ic:
            if if_process and i%500==0:
                t2=time()
                print('CalcLoss-',i,'Time Cost:',round(t2-t1,3))
                t1=time()
            self.Lij[i]={}
            self.Lijc={}
            for j in self.Knn[i]:
                if j==i:
                    continue
                self.Lijc[j],self.Lij[i][j]={},{}
                X=[1]+self.Calc_X(self.db[i,:],self.db[j,:])
                X=np.array(X)
                for c in SelectedAttr:
                    error=abs(self.DModel[j][c]['Model'] @ X)
                    self.Lijc[j][c]=error
                self.Lij[i][j]=sum(self.Lijc[j].values())
            self.Li[i]=self.Update_Lij(i,Ig)

    def DistModel_m(self): 
        for i in range(self.n):
            ADict={}
            self.DModel[i]={}
            for c in self.DOM.keys():
                ADict[c]={'X':np.zeros((math.comb(self.k_T,2)+self.k_T, self.m)),'Y':np.zeros((math.comb(self.k_T,2)+self.k_T, 1))}
                self.DModel[i][c]={}
            for j_id in range(self.k_T):
                j=self.TList[i][j_id]
                X=self.X_Dict[i][j]
                for c in self.DOM.keys():
                    ADict[c]['Y'][j_id]=X[c+1]
                    ADict[c]['X'][j_id,:]=np.concatenate((X[:c+1],X[c+2:]))
            id=self.k_T
            for j_id in range(self.k_T):
                for j_id2 in range(j_id+1,self.k_T):
                    j,j2=self.TList[i][j_id],self.TList[i][j_id2]
                    X=self.X_Dict[j][j2]
                    for c in self.DOM.keys():
                        ADict[c]['Y'][id]=X[c+1]
                        ADict[c]['X'][id,:]=np.concatenate((X[:c+1],X[c+2:]))
                    id+=1
            for c in self.DOM.keys():
                coef,sigma2=TrainDModel_byHand(ADict[c]['X'],ADict[c]['Y'],self.m,self.k_T)
                self.DModel[i][c]['Sigma^2']=sigma2+1e-7             
                coef=coef.T[0]
                self.DModel[i][c]['Model']=np.concatenate((coef[:c+1],-1*np.ones((1,)),coef[c+1:]),axis=0)


    def CalcDomDist_m(self,if_CheckMem=False):      
        for c in range(self.m):                   
            self.DomDist_dict[c]={}
            D=len(self.DOM[c])
            self.val_id[c]={}
            for i in range(D):
                self.val_id[c][self.DOM[c][i]]=i
            self.DomDist[c]=np.zeros((D,D))
            if D==1:
                continue
            if isinstance(self.DOM[c][0],str):     
                dM=0
                for i in range(D):
                    for j in range(i+1,D):
                        len1,len2=len(self.DOM[c][i]),len(self.DOM[c][j])
                        d=Str_Dist_Between_2_Ele(self.DOM[c][i],self.DOM[c][j])
                        self.DomDist[c][i,j]=2*d/(len1+len2+d)
                self.DomDist[c]+=self.DomDist[c].T

            else:
                dM=0
                for i in range(D):
                    for j in range(i+1,D):
                        self.DomDist[c][i,j]=Num_Dist_Between_2_Ele(self.DOM[c][i],self.DOM[c][j])
                        if self.DomDist[c][i,j]>dM:
                            dM=self.DomDist[c][i,j]*1
                self.DomDist[c]/=dM
                self.DomDist[c]+=self.DomDist[c].T


    def CalcTpDist_m(self,if_CheckMem=False):     
        self.TupleDist=np.zeros((self.n,self.n)) 
        for i in range(self.n):
            for c in range(self.m):
                self.db_val_id[i,c]=self.val_id[c][self.db[i,c]]
        for i in range(self.n):
            for j in range(i+1,self.n):
                X=[self.DomDist[c][self.db_val_id[i,c],self.db_val_id[j,c]] for c in range(self.m)]
                self.TupleDist[i,j]=sum(X)
        self.TupleDist+=self.TupleDist.T




def process_i(i, temp, db, Calc_X, Ig, K, k_T):
    TupleDist_i = {}
    for j in temp:
        if i == j:
            continue
        X = Calc_X(db[i, :], db[j, :])
        TupleDist_i[j] = sum(X)

    key_set = sorted(TupleDist_i.keys(), key=lambda x: TupleDist_i[x])
    Knn_i = [x for x in key_set[:K]]

    num_TList = 0
    TList_i = []
    for j in key_set:
        if j == i:
            continue
        if j in Ig:
            TList_i.append(j)
            num_TList += 1
            if num_TList >= k_T:
                break

    return i, Knn_i, TList_i




def process_chunk(chunk, temp, db, Calc_X, Ig, K, k_T, if_process=False):
    chunk_result = []
    total_tasks = len(chunk)
    
    for idx, i in enumerate(chunk):
        TupleDist_i = {}
        for j in temp:
            if i == j:
                continue
            X = Calc_X(db[i, :], db[j, :])
            TupleDist_i[j] = sum(X)

        key_set = sorted(TupleDist_i.keys(), key=lambda x: TupleDist_i[x])
        Knn_i = [x for x in key_set[:K]]

        num_TList = 0
        TList_i = []
        for j in key_set:
            if j == i:
                continue
            if j in Ig:
                TList_i.append(j)
                num_TList += 1
                if num_TList >= k_T:
                    break

        chunk_result.append((i, Knn_i, TList_i))
        
        if if_process:
            if (idx + 1) % 500 == 0:
                progress = (idx + 1) / total_tasks * 100 
                print(f"Process {chunk[0]} - {chunk[-1]}: Processed {idx + 1}/{total_tasks} tasks, {progress:.2f}% complete. Last processed i: {i}")
    
    return chunk_result







