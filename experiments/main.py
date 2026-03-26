import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main_core import *
from copy import deepcopy
import math
import numpy as np
import pandas as pd
import pickle
from util.classifier import Classifier
from util.Assist import *
from util.u_repair_tools import *
import matplotlib.pyplot as plt
import random



if __name__ == '__main__':


    DataSet_Set=["res"]
    MethodSet=[ "Probabilistic" ]   
    LineRange="All"
    task=1
 

    # Experiments: S-repair performance
    if task==1:         
        if_read=False           
        if_read_cf=False
        gamma_=3
        for DataSet in DataSet_Set:
            print('============', DataSet , '============')
            resTable=np.zeros((len(MethodSet),5),dtype=object)
            id=0        
            ls=LS(DataSet=DataSet, LineRange=LineRange,
                if_process=True,if_CheckMem=False,if_read=if_read,if_read_cf=if_read_cf, 
                Max_Turn=5,gamma_=gamma_,
                max_workers=6,regression_method="linear", 
                )
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
        GammaSet=[x*0.2 for x in range(11)] + [5,10,15,20,25,30,35,40]
        resTable_f1=np.zeros((len(MethodSet),len(GammaSet)+1),dtype=object)     
        resTable_t=np.zeros((len(MethodSet),len(GammaSet)+1),dtype=object)  
        DataSet= DataSet_Set[0]
        c=1
        for gamma_ in GammaSet:
            id = 0
            resTable_f1[0,c],resTable_t[0,id]=gamma_,gamma_
            print('============', DataSet , '============')
            ls=LS(DataSet=DataSet, LineRange=LineRange,if_process=True,if_CheckMem=False,if_read=True,if_read_cf=True)
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

    # Sensitivity: G
    if task==5:
        DataSet_Set=['rayyan']
        MethodSet=['Probabilistic','Clique','ILP']  
        G_rate=[0.25,0.5,0.75,1,3,5,7,9]
        resTable_f1=np.zeros((len(MethodSet)+1,len(G_rate)+1),dtype=object)    
        resTable_t=np.zeros((len(MethodSet)+1,len(G_rate)+1),dtype=object) 
        DataSet= DataSet_Set[0]
        c=1
        for g_rate in G_rate:
            id=0
            resTable_f1[0,c],resTable_t[0,id]=g_rate,g_rate
            print('============', DataSet , '============')
            ls=LS(DataSet=DataSet, LineRange=LineRange,if_process=True,if_CheckMem=False,if_read=True,if_read_cf=True,gamma_=3)
            ls.Basis()
            for Method in MethodSet:
                resTable_f1[id,0],resTable_t[id,0]=Method,Method
                _,_,resTable_f1[id,c],resTable_t[id,c]=ls.Core(Method,g_rate=g_rate) 
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



    # Application: Classification
    if task==7:
        train_set_rate=0.5
        pct_set=[5,10,15,20]
        rounds=10
        for DataSet in DataSet_Set:
            print('============', DataSet , '============')
            ResTable=resTable=np.zeros(( len(MethodSet)+1, len(pct_set)+1 ),dtype=object) 
            
            for r in range(rounds):
                print('-------------', f"round : {r}" , '-------------')
                resTable=np.zeros(( len(MethodSet)+1, len(pct_set)+1 ),dtype=object) 
                
                id_pct=1
                for pct in pct_set:
                    resTable[0,id_pct]=pct
                    ResTable[0,id_pct]+=pct
                    ls=LS(DataSet=DataSet, LineRange=LineRange,
                        if_process=True,if_CheckMem=False,if_read=False,if_read_cf=False, 
                        Max_Turn=5,gamma_=1,
                        max_workers=6,regression_method="linear", 
                        downstream=True,pct=pct, )

                    ls.DataLoading() 
                    ls.DataLoading_clean()
                    ls.train_test_split(train_set_rate=train_set_rate)
                    id_method=1
                    for method in MethodSet:
                        resTable[id_method,0]=method
                        ResTable[id_method,0]=method
                        db_train, db_test = ls.Read_repaired(method)
                        try:
                            acc, model, le = train_and_eval_mlp_multiclass(db_train, db_test)
                        except Exception as e:
                            print(e)
                            print(DataSet)
                            print(pct)
                            print(method)
                        resTable[id_method,id_pct]=acc
                        ResTable[id_method,id_pct]+=acc
                        id_method+=1
                    id_pct+=1
            numeric_part = ResTable[1:, 1:].astype(float) / rounds
            numeric_part = np.round(numeric_part, 3)
            ResTable[1:, 1:] = numeric_part

        
            
    # Application: Clustering
    if task==8:
        pct_set=[5,10,15,20]

        for DataSet in DataSet_Set:
            print('============', DataSet , '============')
            resTable=np.zeros(( len(MethodSet)+1, len(pct_set)+1 ),dtype=object) 
            id_pct=1
            for pct in pct_set:
                resTable[0,id_pct]=pct
                
                ls=LS(DataSet=DataSet, LineRange=LineRange,
                    if_process=True,if_CheckMem=False,if_read=False,if_read_cf=False, 
                    Max_Turn=5,gamma_=1,
                    max_workers=6,regression_method="linear",
                    downstream=True,pct=pct, )
                ls.DataLoading() 
                ls.DataLoading_clean()
                id_method=1
                for method in MethodSet:
                    resTable[id_method,0]=method
                    db_repaired, db_truth = ls.Read_repaired3(method)
                    ari = ls.clustering_ari_on_repaired(db_repaired, db_truth, method="kmeans")
                    resTable[id_method,id_pct]=ari
                    id_method+=1
                id_pct+=1



    # Application: Regression
    if task==9:
        train_set_rate=0.5
        rounds=10
        for DataSet in DataSet_Set:
            print('============', DataSet , '============')
            ResTable=resTable=np.zeros(( len(MethodSet)+1, 1+1 ),dtype=object) 
            
            for r in range(rounds):
                print('-------------', f"round : {r}" , '-------------')
                resTable=np.zeros(( len(MethodSet)+1, 1+1 ),dtype=object) 
                resTable[0,1]=pct
                ResTable[0,1]=pct
                ls=LS(DataSet=DataSet, LineRange=LineRange,
                    if_process=True,if_CheckMem=False,if_read=False,if_read_cf=False, 
                    Max_Turn=5,gamma_=1,
                    max_workers=6,regression_method="linear",
                    downstream=True,pct=pct, )
                ls.DataLoading() 
                ls.DataLoading_clean()
                ls.train_test_split(train_set_rate=train_set_rate)
                id_method=1
                for method in MethodSet:
                    resTable[id_method,0]=method
                    ResTable[id_method,0]=method
                    db_train, db_test = ls.Read_repaired(method)
                    try:
                        if DataSet=="AirQuality":
                            target="C6H6(GT)"
                        elif DataSet=="SocialMedia":
                            target="job_satisfaction_score"
                        rmse = train_and_eval_regression(db_train, db_test,target=ls.fh.attr_id[target])
                    except Exception as e:
                        print(e)
                        print(DataSet)
                        print(pct)
                        print(method)
                    resTable[id_method,1]=rmse
                    ResTable[id_method,1]+=rmse
                    id_method+=1
            numeric_part = ResTable[1:, 1:].astype(float) / rounds
            numeric_part = np.round(numeric_part, 3)
            ResTable[1:, 1:] = numeric_part

        



