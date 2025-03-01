import numpy as np
import pandas as pd
from Levenshtein import distance
import heapq
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from math import pi,log,exp,sqrt
import copy
from gurobipy import Model, GRB, LinExpr
from os.path import abspath,dirname
import decimal
import matplotlib.pyplot as plt
import networkx as nx
from pympler import asizeof
from util.classifier import Classifier
import random
import chardet


def Num_Dist_Between_2_Ele(a,b):
    return abs(a-b)

def Str_Dist_Between_2_Ele(a,b):
    return distance(a,b)


def Num_Dist_Between_2_Ele_Sta(a,b,dM):
    return abs(a-b)/dM

def Str_Dist_Between_2_Ele_Sta(a,b):
    return 2*distance(a,b)/(len(a)+len(b)+distance(a,b))


def Calc_X(l1,l2,Col_DM,Col_Type):
    X=[Str_Dist_Between_2_Ele_Sta(l1[c],l2[c]) if c in Col_Type['str'] else Num_Dist_Between_2_Ele_Sta(l1[c],l2[c],Col_DM[c]) for c in range(len(l1))]




def find_top_k(data, k, Type='smallest'):   
    if Type=='smallest':
      top_k = list(heapq.nsmallest(k, data))  
    elif Type=='largest':
      top_k = list(heapq.nlargest(k, data))
    indices = []                          
    tem={}
    for i, x in enumerate(data):
        if x not in tem:
          tem[x]=[i]
        else:
          tem[x].append(i)
    for x in top_k:
        if tem[x][0] not in indices:
           indices+=tem[x]
    indices=indices[:k]
    return top_k, indices       


def TrainDModel_byHand(X,Y,m,k):      
    alpha=1e-2  
    phi=np.linalg.inv(X.T @ X+ alpha * np.eye(m)) @ X.T @ Y
    e=np.zeros((k,))
    for i in range(k):
       e[i]=abs(Y[i]-phi.T @ X[i,:])
    var=np.mean(e)
    return phi,var

def CalcLij(error,sigma2):
    return -log(2*pi*sigma2)/2-error**2/(2*sigma2)

def CalcLij_f(error,sigma2):
    return 1/sqrt(2*pi*sigma2)*exp(-error**2/(2*sigma2))

def pEQ(a,b):
    return a==b
def pIEQ(a,b):
    return a!=b        
def pGEQ(a,b):
    return a>=b  
def pLEQ(a,b):
    return a<=b  
def pG(a,b):
    return a>b  
def pL(a,b):
    return a<b  
fDict={'>':pG,
    '>=':pGEQ,
    '<':pL,
    '<=':pLEQ,
    '==':pEQ,
    '!=':pIEQ}


def FindNan(arr):
    nan_mask = np.isnan(arr)
    nan_indices = np.where(nan_mask)
    return nan_indices

def sgn(x):
    if x>0:
        return 1
    elif x==0:
        return 0
    else:
        return -1






def Gamma(pct,gamma_=2.5):
    r=1
    if pct>0:
        for i in range(1,pct+1):
            r*=(i+gamma_)/i 
    elif pct<0:
        for i in range(1,abs(pct)+1):
            r*=(i+gamma_)/i
        r=1/r
    return r



def B2GB(bytes_value):   
    gb_value = bytes_value / (1024 ** 3)
    return gb_value

def CalcMem(var):           
    mu=asizeof.asizeof(var)    
    print('Memory Cost',B2GB(mu),' GB')


def clf(n,DataSet,pct,IN,largest_turn=10):
    res={('N','g'):[],
        ('g','N'):[],
        ('N','N'):[],
        ('g','g'):[]}
    DataSet_whole=(DataSet+'/'+pct)
    if len(IN)>0:
        for i_turn in range(largest_turn):
            file_path=dirname(dirname(abspath(__file__)))+'/data/'+DataSet_whole+'/dirty.csv'
            Ig=[x for x in range(n) if x not in IN]
            print('------------------------------')
            if DataSet=='iris':
                test_ratio = 0.6
            elif DataSet=='yeast':
                test_ratio = 0.8
            shuffled_IN = random.sample(IN, len(IN))
            test_length_IN = int(len(shuffled_IN) * test_ratio)
            train_length_IN = len(shuffled_IN) - test_length_IN
            IN_train = sorted(set(shuffled_IN[:train_length_IN]))
            IN_test = sorted(set(shuffled_IN[train_length_IN:]))
            shuffled_Ig = random.sample(Ig, len(Ig))
            test_length_Ig = int(len(shuffled_Ig) * test_ratio)
            train_length_Ig = len(shuffled_Ig) - test_length_Ig
            Ig_train = sorted(set(shuffled_Ig[:train_length_Ig]))
            Ig_test = sorted(set(shuffled_Ig[train_length_Ig:]))
            pre=[]
            I_train,I_test=IN_train*1,Ig_test*1      
            for i in range(largest_turn):
                pre.append(Classifier(file_path,I_train,I_test))
            pre=round(sum(pre)/largest_turn,3)
            print(f'(N,g) precision: {pre}')
            res[('N','g')].append(pre)
            pre=[]
            I_train,I_test=Ig_train*1,IN_test*1      
            for i in range(largest_turn):
                pre.append(Classifier(file_path,I_train,I_test))
            pre=round(sum(pre)/largest_turn,3)
            print(f'(g,N) precision: {pre}')
            res[('g','N')].append(pre)
            pre=[]
            I_train,I_test=IN_train*1,IN_test*1
            for i in range(largest_turn):
                pre.append(Classifier(file_path,I_train,I_test))
            pre=round(sum(pre)/largest_turn,3)
            print(f'(N,N) precision: {pre}')
            res[('N','N')].append(pre)
            pre=[]
            I_train,I_test=Ig_train*1,Ig_test*1             
            for i in range(largest_turn):
                pre.append(Classifier(file_path,I_train,I_test))
            pre=round(sum(pre)/largest_turn,3)
            print(f'(g,g) precision: {pre}')
            res[('g','g')].append(pre)
        resAvg={k:sum(res[k])/largest_turn for k in res.keys()}
    else: 
        resAvg={('N','g'):0,
            ('g','N'):0,
            ('N','N'):0,
            ('g','g'):0}
    return res,resAvg

def detect_encoding(file_path):    
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
    return result['encoding']


def Save_IN(ls,Method):
    ls.IN=sorted(ls.IN)
    with open(f'./data/{ls.DataSet}/{Method}_IN.txt', 'w') as file:
        for i in ls.IN:
            file.write(f"{i}\n")
