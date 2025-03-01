import numpy as np
import pandas as pd
import os
import random
from util.FileHandler import FileHandler
from util.DC import DC
from time import time
from util.DataHandler import DataHandler
from algorithm.Probabilistic import Probabilistic
from algorithm.ILP_LP import ILP_LP
from util.ResultAnalysis import ResultAnalysis
from util.Assist import *

class LS():
    def __init__(self,DataSet,K=250,LineRange='All', 
                 fh=0,dc=0,Pair=0,t1_3=0, 
                 gamma=1.2,gamma_=1,
                 if_process=True,if_CheckMem=False,thre=0, if_read=False, if_read_cf=False,only_detectable=False,
                 Max_Turn=1e9,pct=''):
        self.DataSet=DataSet
        self.gamma=gamma
        self.gamma_=gamma_
        self.if_process=if_process
        self.if_CheckMem=if_CheckMem
        self.thre=thre
        self.if_read=if_read
        self.if_read_cf=if_read_cf
        self.Max_Turn=Max_Turn
        self.pct=pct
        self.only_detectable=only_detectable
        self.n=0
        self.if_all_str=False
        self.K=K

        if DataSet=='soccer':
            self.dirtyfile='dirty-0.35_'
            self.DirtyPath=DataSet+'/'+self.dirtyfile+'.csv'
            self.CleanPath='soccer/clean.csv'
            self.k_T=5
            self.k_L=4
            self.M=0.1
            self.index_col=0
            self.LineRange=LineRange

        elif DataSet=='res':
            self.DirtyPath='res/dirty-0.35_.csv'              
            self.CleanPath='res/clean.csv'
            self.k_T=4
            self.k_L=4
            self.M=1.5
            self.index_col=0
            self.LineRange=LineRange

        elif DataSet=='rayyan': 
            self.DirtyPath='rayyan/dirty.csv'  
            self.CleanPath='rayyan/clean.csv'
            self.k_L=4
            self.k_T=5
            self.M=0.1
            self.gamma_=2
            self.index_col=0
            self.LineRange=LineRange

        elif DataSet=='spstock':
            self.DirtyPath='spstock/dirty.csv'
            self.CleanPath='spstock/clean.csv'
            self.k_T=9
            self.k_L=20 
            self.M=3
            self.index_col=0
            self.LineRange=LineRange

        elif DataSet=='yeast':       
            self.DirtyPath=f'/yeast/{self.pct}/dirty_no_label.csv'
            self.CleanPath=f'/yeast/{self.pct}/clean_no_label.csv'
            self.k_T=5
            self.k_L=4
            self.M=0.1
            self.index_col=0 
            self.LineRange=LineRange
        
        elif DataSet=='iris':
            self.DirtyPath=f'/iris/{self.pct}/dirty_no_label.csv'
            self.CleanPath=f'/iris/{self.pct}/clean_no_label.csv'
            self.k_T=5
            self.k_L=4
            self.M=0.1
            self.index_col=0 
            self.LineRange=LineRange
        
        
        elif DataSet=='flights':
            self.DirtyPath='flights/dirty.csv'
            self.CleanPath='flights/clean.csv'
            self.k_T=5
            self.k_L=5
            self.M=0.1
            self.index_col=0
            self.LineRange=LineRange
        

        elif DataSet=='inspection':
            self.DirtyPath=DataSet+'/dirty.csv'
            self.CleanPath=DataSet+'/clean.csv'
            self.k_T=10
            self.k_T2=5
            self.k_L=4
            self.M=0.1
            self.index_col=0 
            self.LineRange=LineRange
            per=1
        else: 
            self.DirtyPath=DataSet+'/dirty.csv'
            self.CleanPath=DataSet+'/clean.csv'
            self.k_T=6
            self.k_L=4
            self.M=0.1
            self.index_col=0
            self.LineRange=LineRange      
             


        self.Pair=Pair
        self.IN=0
        if Pair==0:
            self.ifcopy=False
        else:
            self.ifcopy=True
            self.t1_4=t1_3
        self.fh=fh
        self.dc=dc
        self.dh=0
        self.F=0
        self.fh_clean=0
        self.ra=0



    

    def DataLoading(self):
        self.fh=FileHandler(self.DataSet, LineRange=self.LineRange)
        self.fh.Loader(self.DirtyPath, index_col=self.index_col)
        self.fh.fullna()
        self.fh.AttrId()
        self.n=len(self.fh.db)*1
        self.dc=DC(self.DataSet)
        self.dc.setAttrId(self.fh.attr_id)
        if self.DataSet in {'yeast','iris'}:
            self.dc.LoadCons(file=f'/{self.pct}/CONS.txt')
        else: 
            self.dc.LoadCons(file='CONS.txt')
        if self.if_process:
            print('1-complete')

    def Detection(self,task=0):            
        if not self.if_read_cf and task!=8:
            self.dc.check_cf_num(self.fh.db,if_CheckMem=self.if_CheckMem,if_process=self.if_process) 
            self.dc.Save_CfPair(self.DataSet)
        else: 
            self.dc.Read_CfPair(self.DataSet)
            self.dc.Ic=set()
            for p in self.dc.CfPair:
                self.dc.Ic.update(set(p))
            self.dc.I=set(range(self.n))
            if max(self.dc.Ic)<=self.n:   
                self.dc.Ig={x for x in self.dc.I if x not in self.dc.Ic}
            else: 
                tem={x for x in self.dc.Ic if x>self.n}
                temCf={p for p in self.dc.CfPair}
                for p in temCf:
                    if set(p)&tem:
                        self.dc.CfPair.remove(p)
                self.dc.Ic=set()
                for p in self.dc.CfPair:
                    self.dc.Ic.update(set(p))
                self.dc.Ig={x for x in self.dc.I if x not in self.dc.Ic}
        self.dc.ObjErrorTuple_CoveringEdge2(if_CheckMem=self.if_CheckMem,if_process=self.if_process)     
        if self.if_process:
            print('2-complete')

    def DataHandling_m(self,k):  
        t31=time()
        self.dh=DataHandler(self.fh.db, k_T=self.k_T, k_L=self.k_L)
        t_dist1=time()
        self.dh.DomGenerator()
        self.dh.CalcDomDist(if_CheckMem=self.if_CheckMem,if_process=self.if_process)
        t_dist2=time()
        self.t_dist=t_dist2-t_dist1
        t32=time() 
        if self.if_process:
            print('(3)1-2:',round(t32-t31,3))
        if not self.if_read:
            self.dh.CalcTpDist(self.dc.Ig, self.K, if_CheckMem=self.if_CheckMem,if_process=self.if_process)
            self.dh.Save_Knn_TList(self.DataSet)
        else: 
            self.dh.Read_Knn_TList(self.DataSet)
            if max(self.dh.Knn.keys())>self.n:
                self.dh.Knn_Prune(self.K,  self.dc.Ig)
        t33=time() 
        if self.if_process:
            print('(3)2-3:',round(t33-t32,3)) 
        t34=time()       
        if self.if_process:                   
            print('(3)3-4:',round(t34-t33,3))
        self.dh.DistModel(if_process=self.if_process)          
        t35=time()
        if self.if_process:
            print('(3)4-5:',round(t35-t34,3))
        SelectedAttr=random.sample(range(self.dh.m), k)               
        self.dh.CalcLoss_m2(self.dc.I, self.dc.Ig,SelectedAttr) 
        # self.dh.CalcLoss(self.dc.I, self.dc.Ig, if_CheckMem=self.if_CheckMem,if_process=self.if_process)
        t36=time()                     
        if self.if_process:
            print('(3)5-6:',round(t36-t35,3))     
            print('3-complete')
        self.t_TpDist=t33-t32

    def DataHandling(self):  
        t31=time()
        self.dh=DataHandler(self.fh.db, k_T=self.k_T, k_L=self.k_L)
        t_dist1=time()
        self.dh.DomGenerator()
        self.dh.CalcDomDist(if_CheckMem=self.if_CheckMem,if_process=self.if_process)
        t_dist2=time()
        self.t_dist=t_dist2-t_dist1
        t32=time() 
        if self.if_process:
            print('(3)1-2:',round(t32-t31,3))
        if not self.if_read:
            self.dh.CalcTpDist(self.dc.Ig, self.K, if_CheckMem=self.if_CheckMem,if_process=self.if_process)
            self.dh.Save_Knn_TList(self.DataSet)
        else: 
            self.dh.Read_Knn_TList(self.DataSet)
            if max(self.dh.Knn.keys())>self.n:
                self.dh.Knn_Prune(self.K,  self.dc.Ig)
        t33=time() 
        if self.if_process:
            print('(3)2-3:',round(t33-t32,3)) 
        t34=time()       
        if self.if_process:                   
            print('(3)3-4:',round(t34-t33,3))
        self.dh.DistModel(if_process=self.if_process)          
        t35=time()
        if self.if_process:
            print('(3)4-5:',round(t35-t34,3))
        self.dh.CalcLoss(self.dc.I, self.dc.Ig, if_CheckMem=self.if_CheckMem,if_process=self.if_process)
        t36=time()                     
        if self.if_process:
            print('(3)5-6:',round(t36-t35,3))     
            print('3-complete')
        self.t_TpDist=t33-t32

    def ProbMain(self,max_turn=10):  
        self.F=Probabilistic(self.dc,self.dh,if_KeepIc_obj=False)
        self.F.Pos()
        self.F.Enhancement(gamma_=self.gamma_)                  
        self.fh_clean=FileHandler(self.DataSet,self.LineRange)
        self.fh_clean.Loader(self.CleanPath,index_col=self.index_col)
        ra=ResultAnalysis(db_clean=self.fh_clean.db, db_dirty=self.fh.db, n=self.dh.n, m=self.dh.m)
        ra.S_Repair_GroundTruth()
        resTable={i:{'pr':0, 'rc':0, 'f1':0, 't':0} for i in range(max_turn)}
        IN,fmax=[],0
        for i in range(max_turn):
            t1=time()
            in_=self.F.Main()
            t2=time()
            ra.S_Repair_Changed(in_)
            resTable[i]['t']=self.t1_4+t2-t1 
            resTable[i]['pr'],resTable[i]['rc'],resTable[i]['f1']=ra.S_Repair_Calc_3_metric()
            if resTable[i]['f1']>fmax:
                fmax=resTable[i]['f1']
                IN=in_
        self.IN=IN
        pr_avg=sum(resTable[i]['pr'] for i in range(max_turn))/max_turn
        rc_avg=sum(resTable[i]['rc'] for i in range(max_turn))/max_turn
        f1_avg=sum(resTable[i]['f1'] for i in range(max_turn))/max_turn
        t_avg=sum(resTable[i]['t'] for i in range(max_turn))/max_turn
        resTable2={'pr':round(pr_avg,3), 'rc':round(rc_avg,3), 'f1':round(f1_avg,3), 't':round(t_avg,3)}
        return round(pr_avg,3),round(rc_avg,3),round(f1_avg,3),round(t_avg,3)

    def LpMain(self):   
        self.F=ILP_LP(self.dc.Ic_obj, self.dh.Lij,  k_L=self.k_L, if_KeepIc_obj=False, Cf_Set=self.dc.Cf_Set)
        self.F.Pos()
        self.F.Enhancement(gamma_=self.gamma_)
        self.IN=self.F.Solve_with_Clique(if_CheckMem=self.if_CheckMem,Max_Turn=self.Max_Turn)       
        self.IN=self.F.Minimization(self.IN)  
        return self.IN

    def ILpMain(self): 
        self.F=ILP_LP(self.dc.Ic_obj, self.dh.Lij,  k_L=self.k_L, if_KeepIc_obj=False, Cf_Set=self.dc.Cf_Set)
        self.F.Pos()
        self.F.Enhancement(gamma_=self.gamma_)  
        self.IN=self.F.LP_Solver(if_binary=True,if_CheckMem=self.if_CheckMem)       
        return self.IN
    
    def Result_Analysis(self):
        self.fh_clean=FileHandler(self.DataSet,self.LineRange)
        self.fh_clean.Loader(self.CleanPath, index_col=self.index_col)
        self.ra=ResultAnalysis(db_clean=self.fh_clean.db, db_dirty=self.fh.db, n=self.dh.n, m=self.dh.m)
        self.ra.S_Repair_GroundTruth()
        self.ra.S_Repair_Changed(self.IN)
        self.precision, self.recall, self.f1=self.ra.S_Repair_Calc_3_metric()
       
    def Basis_m(self):       
        t1=time()
        self.DataLoading()
        t2=time()
        self.Detection()
        t3=time()
        self.t1_3=t3-t1

    def Basis(self,task=0):       
        t1=time()
        self.DataLoading()
        t2=time()
        self.Detection(task)
        t3=time()
        if self.if_process:
            print('Detection time:',round(t3-t2,3))
        self.DataHandling()           
        t4=time()
        if self.if_process:
            print('Handling time:',round(t4-t3))
        self.t1_4=t4-t1
        self.t1_4=self.t1_4-self.t_TpDist
        self.t1_3=t3-t1
        return self.t1_4

    def Core(self,Method):  
        self.Method=Method
        print('===========================================')
        print('----------    ',self.Method,'    ----------')
        t4=time()
        if self.n>self.thre: 
            if self.Method=='Probabilistic':
                max_turn=10
                self.precision,self.recall,self.f1,t=self.ProbMain(max_turn=max_turn)
                t_end=time()
                t=(t_end-t4)/max_turn
                t_total=t      
                self.F=None
            elif self.Method=='Clique':
                self.IN=self.LpMain()
                self.Result_Analysis()
                t_end=time()
                t_total=t_end-t4   
                self.F=None
            elif self.Method=='ILP':
                self.F=ILP_LP(self.dc.Ic_obj, self.dh.Lij,  k_L=self.k_L, if_KeepIc_obj=False, Cf_Set=self.dc.Cf_Set)
                self.IN=self.ILpMain()
                self.Result_Analysis()
                t_end=time()
                t_total=t_end-t4     
                self.F=None
        t_total=round(t_total,3)
        print('Time:',t_total) 
        return self.precision, self.recall, self.f1, t_total

    def Core_m(self,Method,k): 
        t3=time()
        self.DataHandling_m(k)
        t4=time()
        self.t1_4=self.t1_3+t4-t3
        if Method=='Probabilistic':
            max_turn=10
            self.precision, self.recall, self.f1, t=self.ProbMain(max_turn=max_turn)
            t_end=time()
            t=(t_end-t4)/max_turn
            return self.precision, self.recall, self.f1,t
        elif Method=='Clique':
            self.IN=self.LpMain()
            self.Result_Analysis()
            t_end=time()
            t=t_end-t4
            return self.precision, self.recall, self.f1,t
        elif Method=='ILP':
            self.IN=self.ILpMain()
            self.Result_Analysis()
            t_end=time()
            t=t_end-t4
            return self.precision, self.recall, self.f1,t


