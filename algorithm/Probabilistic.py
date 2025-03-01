import numpy as np
import pandas as pd
import random
from math import *
from copy import deepcopy
from util.Assist import *
import os

class Probabilistic():
    def __init__(self,dc,dh,if_Loss=True,if_KeepIc_obj=True):
        self.I=dc.I
        self.Ic=dc.Ic
        self.Ig=dc.Ig
        if if_KeepIc_obj:
            self.Ic_obj=dc.Ic_obj         
        else: 
            self.Cf_Set=dc.Cf_Set
        self.Ig_obj={}  
        self.Lij=dh.Lij
        self.Li=dh.Li
        if dh.LList=={}:
            if if_Loss:
                dh.RelationL_byLoss(self.I)
            else:
                dh.RelationL(self.I)
        self.LList=deepcopy(dh.LList)
        self.k_L=dh.k_L
        self.Lpi,self.Lpij={i:'a' for i in self.I}, {i:{} for i in self.I}
        self.if_Loss=if_Loss


    def Main(self):   
        IN={x for x in self.Ic if x in self.Cf_Set[x]}
        for i in self.Ic:
            for j in self.Cf_Set[i]:
                if j<=i:
                    continue 
                if self.Li[i]<1e-8 and self.Li[j]<1e-8:   
                    p=0.5
                else:
                    p=self.Li[j]/(self.Li[i]+self.Li[j])
                IN.add(random.choices((i,j), k=1, weights=[p, 1 - p])[0])

        Ic_=set(self.Ic)-IN   
        IN_=sorted(IN, key=lambda x:self.Li[x],reverse=True)
        for i in IN_:
            if not self.Cf_Set[i] & Ic_:
                Ic_.add(i)
                IN.remove(i)
        return IN

    def Pos(self):     
        tem=[self.Lij[i][j] for i in self.Lij for j in self.Lij[i]]
        Lm,LM=min(tem),max(tem)
        self.Lij={i:{j: LM-self.Lij[i][j] for j in self.Lij[i].keys()} for i in self.Lij}
        self.Li={i:sum(find_top_k(self.Lij[i].values(), self.k_L, Type='largest')[0]) for i in self.Lij.keys()}

    def Enhancement(self,gamma_=2.5):      
        self.Li={i:sum(find_top_k(self.Lij[i].values(), self.k_L, Type='largest')[0]) for i in self.Lij.keys()}
        self.pct={i:0 for i in self.Ic}
        for i in self.Ic:
            for j in self.Cf_Set[i]:
                if j<=i:
                    continue
                if self.Li[i]>self.Li[j]:
                    self.pct[i]+=1
                    self.pct[j]-=1
                elif self.Li[i]<self.Li[j]:
                    self.pct[i]-=1
                    self.pct[j]+=1
        self.pct={i:Gamma(self.pct[i],gamma_=gamma_) for i in self.Ic}
        for i in self.Ic:
            self.Li[i]*=self.pct[i]
