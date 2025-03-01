from main_core import LS
from copy import deepcopy
import math
import numpy as np
import pandas as pd
import os
from util.classifier import Classifier
from util.Assist import *






DataSet_Set=['res']
MethodSet=['Probabilistic','Clique']
LineRange='All'
task=8





# Experiments: S-repair performance
if task==1:         
    if_read=False           
    if_read_cf=False
    gamma_=5
    for DataSet in DataSet_Set:
        print('============', DataSet , '============')
        resTable=np.zeros((len(MethodSet),5),dtype=object)
        id=0        
        ls=LS(DataSet=DataSet, LineRange=LineRange,
              if_process=True,if_CheckMem=False,if_read=if_read,if_read_cf=if_read_cf,
              Max_Turn=5,gamma_=gamma_)
        ls.Basis()
        for Method in MethodSet:
            resTable[id,0]=Method
            resTable[id,1],resTable[id,2],resTable[id,3],resTable[id,4]=ls.Core(Method)
            id,ls.F,ls.IN=id+1,None,None
        resTable=pd.DataFrame(resTable,columns=['','Pre','Rec','F1','Time'])
        print(resTable)
        ls=None

# Sensitivity: \gamma
if task==2:      
    DataSet_Set=['rayyan']       
    GammaSet=[0,2,4,6,8,10,20,30,40,50]
    resTable_f1=np.zeros((len(MethodSet),len(GammaSet)+1),dtype=object)     
    resTable_t=np.zeros((len(MethodSet),len(GammaSet)+1),dtype=object)  
    DataSet= DataSet_Set[0]
    c=1
    for gamma_ in GammaSet:
        id = 0
        resTable_f1[0,c],resTable_t[0,id]=gamma_,gamma_
        print('============', DataSet , '============')
        ls=LS(DataSet=DataSet, LineRange=LineRange,if_process=True,if_CheckMem=False,if_read=False)
        ls.gamma_=gamma_
        ls.Basis()
        for Method in MethodSet:
            resTable_f1[id,0],resTable_t[id,0]=Method,Method
            _,_,resTable_f1[id,c],resTable_t[id,c]=ls.Core(Method)
            id,ls.F,ls.IN=id+1,None,None
        c+=1
    print('Results-F1:')
    print(resTable_f1)
    print('Results-time:')
    print(resTable_t)
    ls=None

# Sensitivity: k
if task==3:         
    DataSet_Set=['rayyan']       
    K_L_Set=[2,4,6,8,10,20,30,40,50]
    resTable_f1=np.zeros((len(MethodSet)+1,len(K_L_Set)+1),dtype=object)   
    resTable_t=np.zeros((len(MethodSet)+1,len(K_L_Set)+1),dtype=object)      
    DataSet= DataSet_Set[0]
    c=1
    for k_L in K_L_Set:
        id = 0
        resTable_f1[0,c],resTable_t[0,id]=k_L,k_L
        print('============', DataSet , '============')
        ls=LS(DataSet=DataSet, LineRange=LineRange,if_process=True,if_CheckMem=False,if_read=False,gamma_=5)
        ls.k_L=k_L
        ls.Basis()
        for Method in MethodSet:
            resTable_f1[id,0],resTable_t[id,0]=Method,Method
            _,_,resTable_f1[id,c],resTable_t[id,c]=ls.Core(Method) 
        id,ls.F,ls.IN=id+1,None,None
        c+=1
    print('Results-F1:')
    print(resTable_f1)
    print('Results-time:')
    print(resTable_t)
    ls=None

# Sensitivity: \kappa
if task==4:             
    DataSet_Set=['rayyan']
    MethodSet=['Probabilistic','Clique','ILP']     
    K_T_Set=list(range(4,11))
    resTable_f1=np.zeros((len(MethodSet)+1,len(K_T_Set)+1),dtype=object)    
    resTable_t=np.zeros((len(MethodSet)+1,len(K_T_Set)+1),dtype=object)     
    DataSet= DataSet_Set[0]
    c=1
    for k_T in K_T_Set:
        id = 0
        resTable_f1[0,c],resTable_t[0,id]=k_T,k_T
        print('============', DataSet , '============')
        ls=LS(DataSet=DataSet, LineRange=LineRange,if_process=True,if_CheckMem=False,if_read=False,gamma_=5)
        ls.k_T=k_T
        ls.Basis()
        for Method in MethodSet:
            resTable_f1[id,0],resTable_t[id,0]=Method,Method
            _,_,resTable_f1[id,c],resTable_t[id,c]=ls.Core(Method)    
            id,ls.F,ls.IN=id+1,None,None
        c+=1
    print('Results-F1:')
    print(resTable_f1)
    print('Results-time:')
    print(resTable_t)
    ls=None

# Sensitivity: m
if task==6:             
    DataSet='rayyan'
    MethodSet=['Probabilistic','Clique','ILP']
    repite=10
    ls=LS(DataSet=DataSet, LineRange=LineRange, gamma_=5, Max_Turn=15)
    ls.Basis_m()
    m=ls.fh.db.shape[1]
    resdict={Method:{k:{i:{} for i in range(repite)} for k in range(1,m+1)} for Method in MethodSet}
    resdict_avg={Method:{k:{'f1':0,'t':0} for k in range(1,m+1)} for Method in MethodSet}
    for k in range(1,m+1):
        print('==========',k,'==========')
        for Method in MethodSet:
            for r in range(repite):
                _,_,resdict[Method][k][r]['f1'],resdict[Method][k][r]['t']=ls.Core_m(Method,k)
                ls.dh,ls.F,ls.ra=None,None,None
            resdict_avg[Method][k]={'f1':round(sum(resdict[Method][k][x]['f1'] for x in range(repite))/repite,3),
                                    't':round(sum(resdict[Method][k][x]['t'] for x in range(repite))/repite,3)}
    # Result Table
    resTable_f1=np.zeros((len(MethodSet),m+1),dtype=object)     
    resTable_t=np.zeros((len(MethodSet),m+1),dtype=object)    
    resTable_f1[0,0],resTable_f1[1,0],resTable_f1[2,0]='Probabilistic','Clique','ILP'
    resTable_t[0,0],resTable_t[1,0],resTable_t[2,0]='Probabilistic','Clique','ILP'
    for i in range(3):
        for j in range(1,m+1):
            Method=MethodSet[i]
            resTable_f1[i,j]=resdict_avg[Method][j]['f1']
            resTable_t[i,j]=resdict_avg[Method][j]['t']
    col=list(range(0,m+1))
    print('Results-F1:')
    print(resTable_f1)
    print('Results-time:')
    print(resTable_t)

# Experiments: Classification
if task==7:             
    for pct in ['5%','10%','15%','20%']:
        DataSet='yeast'
        MethodSet=['Probabilistic','Clique']
        largest_turn=5      
        print('============', DataSet ,pct ,'============')
        resTable=np.zeros((len(MethodSet),5),dtype=object)
        id=0        
        ls=LS(DataSet=DataSet, LineRange=LineRange,
            if_process=True,if_CheckMem=False,if_read=False,if_read_cf=False,
            Max_Turn=5,gamma_=5,pct=pct)
        ls.Basis()     
        resAvg={}
        for Method in MethodSet:
            ls.Core(Method)   
            id,ls.F=id+1,None
            res,resAvg[Method]=clf(ls.n, DataSet,pct, ls.IN, largest_turn=largest_turn)

        # Result Table
        resTable=np.zeros((len(MethodSet),5),dtype=object)
        i=0
        for m in resAvg.keys():
            resTable[i,0]=m
            j=1
            for k in resAvg[m].values():
                resTable[i,j]=round(k,3)
                j+=1
            i+=1
            print(m,resAvg[m])
        col=['Method','(N,g)','(g,N)','(N,N)','(g,g)']
        resTable=pd.DataFrame(resTable,columns=col)
        print('Result:')
        print(resTable)









