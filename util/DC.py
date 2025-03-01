import numpy as np
import pandas as pd
from util.Assist import *
from util.txt2cons import txt2cons
import random
from pympler import asizeof
from time import time
from os.path import abspath,dirname
import os


class DC():
    def __init__(self,dataName):
        self.dataName=dataName 
        self.cons={}           
        self.attr_id={}        
        self.if_symme={}       
        self.CfPair={}         
        self.TupleNum={}       
        self.Ic=set()
        self.Ig=set()
        self.I=set()
        self.Ic_obj={}         
        self.ConsAttr=set()   

    def setAttrId(self,attr_id):  
        self.attr_id=attr_id



    def LoadCons(self,file='CONS.txt',path=dirname(dirname(abspath(__file__)))+'/data/',if_symm=False):       
        self.cons=txt2cons(path=(path+self.dataName+'/'+file),if_print=False)
        if not if_symm:
            self.if_symme,self.TupleNum={k:0 for k in self.cons.keys()},{k:2 for k in self.cons.keys()}
        else:
            self.if_symme,self.TupleNum={k:1 for k in self.cons.keys()},{k:2 for k in self.cons.keys()}
        for k in self.cons.keys():         
            for p in self.cons[k]:
                if len(p)==4:             
                    p[1]=self.attr_id[p[1]]
                else:
                    p[1]=self.attr_id[p[1]]
                    p[4]=self.attr_id[p[4]]

    def check_relation(self,x,operator,y):
        if operator=='>':
            if x>y:
                return 1
            else:
                return 0
        elif operator=='>=':
            if x>=y:
                return 1
            else:
                return 0
        elif operator=='==':
            if x==y:
                return 1
            else:
                return 0
        elif operator=='!=':
            if x!=y:
                return 1
            else:
                return 0
        elif operator=='<=':
            if x<=y:
                return 1
            else:
                return 0
        elif operator=='<':
            if x<y:
                return 1
            else:
                return 0

    def ObjErrorTuple_CoveringEdge(self,if_CheckMem=False):  
        Cf_Set={}
        for i in self.Ic:
            Cf_Set[i]=[]
        Pair={}
        j=0
        P1=[p for p in self.CfPair if len(p)==1]
        P2=[p for p in self.CfPair if len(p)==2]
        self.CfPair=None
        for p in P1:
            Cf_Set[p[0]].append(p[0])
            Pair[j]=p
            j+=1
        for p in P2:
            Cf_Set[p[0]].append(p[1])
            Cf_Set[p[1]].append(p[0]) 
            Pair[j]=p
            j+=1
        for i in self.Ic:
            self.Ic_obj[i].Cf_Set=set(Cf_Set[i])
        return Pair

    def ObjErrorTuple_CoveringEdge2(self,if_CheckMem=False,if_process=False): 
        Cf_Set={}
        self.Ic_obj={}
        for i in self.Ic:
            Cf_Set[i]=[]
        j=0
        P1=[p for p in self.CfPair if len(p)==1]
        P2=[p for p in self.CfPair if len(p)==2]
        self.CfPair=None
        temN=len(P1)+len(P2)
        for p in P1:
            Cf_Set[p[0]].append(p[0])
            j+=1
            if j%20000==0 and if_process:
                print('Updated Pairs:',round(j/temN,3))
        for p in P2:
            Cf_Set[p[0]].append(p[1])
            Cf_Set[p[1]].append(p[0]) 
            j+=1
            if j%20000==0 and if_process:
                print('Updated Pairs:',round(j/temN,3))
        self.Cf_Set={i:set(Cf_Set[i]) for i in Cf_Set}





    def check_cf_num(self,db,if_CheckMem=False,if_process=False):      
        res=set()


        D1={d:self.cons[d] for d in self.cons.keys() if self.TupleNum[d]==1}                               
        D2_0={d:self.cons[d] for d in self.cons.keys() if self.TupleNum[d]==2 and self.if_symme[d]==0}    
        D2_fd={}      
        D2_dc={}     
        D3={d:self.cons[d] for d in self.cons.keys() if self.TupleNum[d]==1.5}           
        for d in self.cons.keys():
            if d in D1 or d in D2_0 or d in D3:
                continue
            opeList=[p[2] for p in self.cons[d]]
            if all(ope=='==' for ope in opeList[:-1]) and opeList[-1]=='!=':
                D2_fd[d]=self.cons[d]
            else:
                D2_dc[d]=self.cons[d]
        n,m=db.shape
        t1=time()
        for i in range(n):
            if i%500==0 and if_process:
                t2=time()
                t=round(t2-t1,3)
                print('Check Cf:',i,'Time Cost:',t)
                t1=time()
            for d in D1.keys():
                ifCF=True
                for p in D1[d]:
                    if not fDict[p[2]](db[i,p[1]],p[3]):
                        ifCF=False
                        break
                if ifCF:
                    res.add((i,))


        
            for j in range(i+1,n):
                out=0    
                for d in D2_fd.keys():
                    ifCF=True
                    lenDC=len(D2_fd[d])
                    for pid in range(lenDC):
                        p=D2_fd[d][pid]
                        if pid<lenDC-1 and not db[i,p[1]]==db[j,p[4]]:
                            ifCF=False
                            break
                        elif pid==lenDC-1 and not db[i,p[1]]!=db[j,p[4]]:
                            ifCF=False
                            break
                    if ifCF:
                        res.add((i,j))
                        out=1
                        break
                if out==1:
                    continue
                for d in D2_dc.keys():
                    ifCF=True
                    for p in D2_dc[d]:
                        if not fDict[p[2]](db[i,p[1]],db[j,p[4]]):
                            ifCF=False
                            break
                    if ifCF:
                        res.add((i,j))
                        out=1
                        break
                if out==1:
                    continue
            for j in range(n):
                if i==j or (i,j) in res or (j,i) in res:
                    continue
                out=0
                for d in D2_0.keys():
                    ifCF=True
                    for p in D2_0[d]:
                        if not fDict[p[2]](db[i,p[1]],db[j,p[4]]):
                            ifCF=False
                            break
                    if ifCF:
                        res.add((i,j))
                        out=1
                        break
                if out==1:
                    continue
                for d in D3.keys():
                    ifCF=True
                    for p in D3[d]:
                        if len(p)==5:
                            if not fDict[p[2]](db[i,p[1]],db[j,p[4]]):
                                ifCF=False
                                break
                        elif len(p)==4:
                            l=[i,j]
                            t=l[p[0]-1]
                            if not fDict[p[2]](db[t,p[1]],p[3]):
                                ifCF=False
                                break
                    if ifCF:
                        res.add((i,j))
                        break

        self.CfPair=res
        self.I=set(range(len(db)))
        for p in self.CfPair:
            self.Ic.update(set(p))

        self.Ig=self.I-self.Ic
        self.I=list(self.I)
        self.Ig=list(self.Ig)
        self.Ic=list(self.Ic)





    def Save_CfPair(self,DataSet):     
        if DataSet not in {'yeast','iris'}: 
            path=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+'/data/'+DataSet+'/CfPair.txt'
            with open(path, 'w') as file:
                for pair in self.CfPair:
                    if len(pair) == 1:
                        file.write(f"{pair[0]},None\n")
                    else:
                        file.write(f"{pair[0]},{pair[1]}\n")
    
    def Read_CfPair(self,DataSet):        
        self.CfPair = set()
        path=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+'/data/'+DataSet+'/CfPair.txt'
        with open(path, 'r') as file:
            for line in file:
                parts = line.strip().split(',')
                if parts[1] == 'None':
                    self.CfPair.add((int(parts[0]),))
                else:
                    self.CfPair.add((int(parts[0]), int(parts[1])))













