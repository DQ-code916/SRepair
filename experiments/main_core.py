import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import random
from util.FileHandler import FileHandler
from util.DC import DC
from time import time
from util.DataHandler import DataHandler
from algorithm.Probabilistic import Probabilistic
from algorithm.ILP_LP import ILP_LP
from util.ResultAnalysis import ResultAnalysis
from util.Assist import *
from util.u_repair_tools import *
from collections import defaultdict
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.metrics import adjusted_rand_score


class LS():
    def __init__(self,DataSet,K=250,LineRange='All', 
                 fh=0,dc=0,Pair=0,t1_3=0, 
                 gamma=1.2,gamma_=1,
                 if_process=True,if_CheckMem=False,thre=0, if_read=False, if_read_cf=False,only_detectable=False,
                 Max_Turn=1e9,pct=None,sampling_threshold=1e5,max_workers=6,regression_method="linear",
                 downstream=False):
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
        self.sampling_threshold=sampling_threshold
        self.max_workers=max_workers
        self.regression_method=regression_method            # linear, gauss, tree
        self.g_rate=None

        if DataSet=='soccer':
            # self.dirtyfile='dirty-0.1_'
            self.dirtyfile='dirty'
            self.DirtyPath=DataSet+'/'+self.dirtyfile+'.csv'
            self.CleanPath='soccer/clean.csv'
            self.k_T=5
            self.k_L=4
            self.M=0.1
            self.index_col=0
            self.LineRange=LineRange

        elif DataSet=='res':
            # self.DirtyPath='res/dirty-0.1_.csv'  
            self.DirtyPath='res/dirty.csv'              
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
        elif DataSet=='flights':       
            self.DirtyPath='flights/dirty.csv'
            self.CleanPath='flights/clean.csv'
            self.k_T=5
            self.k_L=4
            self.M=0.1
            self.index_col=0
            self.LineRange=LineRange
        elif DataSet=='Company':       
            self.DirtyPath='Company/dirty.csv'
            self.CleanPath='Company/clean.csv'
            self.k_T=6
            self.k_L=5
            self.M=0.1
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
            if downstream==False: 
                self.DirtyPath=f'/yeast/{self.pct}%/dirty_no_label.csv'
                self.CleanPath=f'/yeast/{self.pct}%/clean_no_label.csv'
                self.index_col=0 
            else: 
                self.DirtyPath=f'/yeast/{self.pct}%/dirty.csv'
                self.CleanPath=f'/yeast/{self.pct}%/clean.csv'
                self.index_col=None
            self.k_T=5
            self.k_L=4
            self.M=0.1
            self.LineRange=LineRange
        
        elif DataSet=='iris':
            if downstream==False: 
                self.DirtyPath=f'/iris/{self.pct}%/dirty_no_label.csv'
                self.CleanPath=f'/iris/{self.pct}%/clean_no_label.csv'
                self.index_col=0 
            else: 
                self.DirtyPath=f'/iris/{self.pct}%/dirty.csv'
                self.CleanPath=f'/iris/{self.pct}%/clean.csv'
                self.index_col=None
            self.k_T=5
            self.k_L=4
            self.M=0.1
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

        elif DataSet=='NYC Parking':
            self.DirtyPath=DataSet+'/dirty.csv'
            self.CleanPath=DataSet+'/clean.csv'
            self.k_T=10
            self.k_T2=5
            self.k_L=4
            self.M=0.1
            self.index_col=0 
            self.LineRange=LineRange
        elif DataSet=="ACS Income":
            self.DirtyPath=DataSet+'/dirty.csv'
            self.CleanPath=DataSet+'/clean.csv'
            self.k_T=8
            self.k_L=4
            self.M=0.1
            self.index_col=0
            self.LineRange=LineRange      
        else: 
            self.DirtyPath=DataSet+'/dirty.csv'
            self.CleanPath=DataSet+'/clean.csv'
            self.k_T=8
            self.k_L=4
            self.M=0.1
            self.index_col=0
            self.LineRange=LineRange    
             
        if regression_method in {"tree"}:
            self.k_T*=5

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
        self.n,self.m=self.fh.db.shape
        self.dc=DC(self.DataSet)
        self.dc.setAttrId(self.fh.attr_id)
        if self.DataSet in {'yeast','iris'}:
            self.dc.LoadCons(file=f'/{self.pct}%/CONS.txt')
        else: 
            self.dc.LoadCons(file='CONS.txt')
        if self.if_process:
            print('1-complete')
    

    def DataLoading_clean(self):
        self.fh_clean=FileHandler(self.DataSet, LineRange=self.LineRange)
        self.fh_clean.Loader(self.CleanPath, index_col=self.index_col)
        self.fh_clean.fullna()
        self.fh_clean.AttrId()
        self.n,self.m=self.fh_clean.db.shape
        self.dc=DC(self.DataSet)
        self.dc.setAttrId(self.fh_clean.attr_id)
        if self.DataSet in {'yeast','iris'}:
            self.dc.LoadCons(file=f'/{self.pct}%/CONS.txt')
        else: 
            self.dc.LoadCons(file='CONS.txt')
        if self.if_process:
            print('1-complete')



    def train_test_split(self, train_set_rate):
        self.train_set=random.sample(range(self.n),int(train_set_rate*self.n))
        self.test_set=[i for i in range(self.n) if i not in self.train_set]

    # For Classification
    def Read_repaired(self,method):
        if method in ['Probabilistic', 'Clique', "TE", ]:
            IN=Read_IN(f"data/{self.DataSet}/{self.pct}%/IN_results/{method}_IN.txt")
            train_set=[i for i in self.train_set if i not in IN]
            test_set=[i for i in self.test_set if i not in IN]
            db_train=self.fh.db[train_set,1:]
            db_test=self.fh.db[test_set,1:]
        return db_train,db_test


    # For Clustering
    def Read_repaired3(self,method):
        if method in ['Probabilistic', 'Clique', "TE", ]:
            IN=Read_IN(f"data/{self.DataSet}/{self.pct}%/IN_results/{method}_IN.txt")
            Ig=[i for i in range(self.n) if i not in IN]
            db_repaired=self.fh.db[Ig,:]
            db_truth=db_repaired[:,1:]*1
        db_repaired = db_repaired[:,1:-1]
        
        return db_repaired,db_truth


    # Clustering Evaluation
    def kmeans_ari_on_repaired(self, db_repaired, db_truth, random_state=0):
        db = db_truth
        y_true = db[:, -1]


        k = len(np.unique(y_true))
        X = np.asarray(db_repaired, dtype=float)
        kmeans = KMeans(
            n_clusters=k,
            random_state=random_state,
            n_init=10
        )
        y_pred = kmeans.fit_predict(X)
        ari = adjusted_rand_score(y_true, y_pred)
        ari = round(float(ari),3)
        return ari


    def clustering_ari_on_repaired(self,
        db_repaired,
        db_truth,
        method="kmeans",
        random_state=0
    ):
        y_true = db_truth[:, -1]
        k = len(np.unique(y_true))
        X = np.asarray(db_repaired, dtype=float)
        if method == "kmeans":
            model = KMeans(
                n_clusters=k,
                random_state=random_state,
                n_init=10
            )
            y_pred = model.fit_predict(X)

        elif method == "hierarchical":
            model = AgglomerativeClustering(
                n_clusters=k,
                linkage="ward"
            )
            y_pred = model.fit_predict(X)

        elif method == "spectral":
            model = SpectralClustering(
                n_clusters=k,
                random_state=random_state,
                assign_labels="kmeans",
                affinity="nearest_neighbors"
            )
            y_pred = model.fit_predict(X)

        else:
            raise ValueError(f"Unknown clustering method: {method}")

        ari = adjusted_rand_score(y_true, y_pred)

        return round(float(ari), 3)


    def Detection(self,task=0):            
        if not self.if_read_cf and task!=8:
            self.dc.check_cf_num(self.fh.db,  self.gt(),  if_CheckMem=self.if_CheckMem,  if_process=self.if_process, max_workers=self.max_workers)
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
            self.dh.CalcTpDist_m(self.dc.Ig, self.K, if_CheckMem=self.if_CheckMem,if_process=self.if_process)
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
        self.dh.CalcLoss_m(self.dc.I, self.dc.Ig,SelectedAttr) 
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
            self.dh.CalcTpDist(self.dc.Ig, self.K, if_CheckMem=self.if_CheckMem, if_process=self.if_process,  max_workers=self.max_workers, sampling_threshold=self.sampling_threshold)
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

    def calc_CV(self):
        self.F=Probabilistic(self.dc,self.dh,if_KeepIc_obj=False)
        self.F.Pos(g_rate=self.g_rate)
        self.F.Enhancement(gamma_=self.gamma_) 
        CV_list=[]
        for i in self.F.Cf_Set.keys():
            for j in self.F.Cf_Set[i]:
                temp=max((self.F.Li[i],self.F.Li[j]))/min((self.F.Li[i],self.F.Li[j]))-1
                CV_list.append(temp)
        return np.std(CV_list, ddof=1)/np.mean(CV_list)



    def ProbMain(self,max_turn=10):  
        self.F=Probabilistic(self.dc,self.dh,if_KeepIc_obj=False)
        self.F.Pos(g_rate=self.g_rate)
        self.F.Enhancement(gamma_=self.gamma_)
        self.fh_clean=FileHandler(self.DataSet,self.LineRange)
        self.fh_clean.Loader(self.CleanPath,index_col=self.index_col)
        self.extra_process_db()
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
        self.F=ILP_LP(self.dc.Ic_obj, self.dh.Lij,  k_L=self.k_L, if_KeepIc_obj=False, Cf_Set=self.dc.Cf_Set, max_workers=self.max_workers,g_rate=self.g_rate)
        self.F.Pos()
        self.F.Enhancement(gamma_=self.gamma_)
        self.IN=self.F.Solve_with_Clique(if_CheckMem=self.if_CheckMem,Max_Turn=self.Max_Turn)       
        self.IN=self.F.Minimization(self.IN)  
        return self.IN

    def ILpMain(self): 
        self.F=ILP_LP(self.dc.Ic_obj, self.dh.Lij,  k_L=self.k_L, if_KeepIc_obj=False, Cf_Set=self.dc.Cf_Set, max_workers=self.max_workers)
        self.F.Pos()
        self.F.Enhancement(gamma_=self.gamma_)  
        self.IN=self.F.LP_Solver(if_binary=True,if_CheckMem=self.if_CheckMem)       
        return self.IN
    
    def Result_Analysis(self):
        self.gt()
        self.ra.S_Repair_Changed(self.IN)
        self.precision, self.recall, self.f1=self.ra.S_Repair_Calc_3_metric()
    

    def gt(self):
        self.fh_clean=FileHandler(self.DataSet,self.LineRange)
        self.fh_clean.Loader(self.CleanPath, index_col=self.index_col)
        self.extra_process_db()
        self.ra=ResultAnalysis(db_clean=self.fh_clean.db, db_dirty=self.fh.db, n=self.n, m=self.m)
        self.ra.S_Repair_GroundTruth()
        print("|error|=",len(self.ra.TrueError))
        return self.ra.TrueError



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


    def Basis_soft(self):       
        t1=time()
        self.DataLoading()
        t2=time()
        self.Detection_soft()    
        t3=time()
        self.t1_3=t3-t1


    def Basis_Downstream(self,task=0):       
        t1=time()
        self.DataLoading()    
        t4=time()
        self.t1_4=t4-t1
        self.t1_4=self.t1_4-self.t_TpDist
        return self.t1_4



    def Core(self,Method,g_rate=None):  
        self.g_rate=g_rate
        self.Method=Method
        print('===========================================')
        print('----------    ',self.Method,'    ----------')
        t4=time()

        if self.Method=='Probabilistic':
            max_turn=10
            self.precision,self.recall,self.f1,t=self.ProbMain(max_turn=max_turn)
            t_end=time()
            t=(t_end-t4)/max_turn
            t_total=t      
            self.F=None

            Save_IN_inner(IN=self.IN,
                          Dataset=self.DataSet,
                          Method=Method,
                          pct=self.pct)
        elif self.Method=='Clique':
            self.IN=self.LpMain()
            self.Result_Analysis()
            t_end=time()
            t_total=t_end-t4   
            self.F=None
            Save_IN_inner(IN=self.IN,
                          Dataset=self.DataSet,
                          Method=Method,
                          pct=self.pct)
        elif self.Method=='ILP':
            self.F=ILP_LP(self.dc.Ic_obj, self.dh.Lij,  k_L=self.k_L, if_KeepIc_obj=False, Cf_Set=self.dc.Cf_Set, max_workers=self.max_workers,g_rate=self.g_rate)
            self.IN=self.ILpMain()
            self.Result_Analysis()
            t_end=time()
            t_total=t_end-t4     
            self.F=None


        t_total=round(t_total,3)
        print("-"*50)
        print("self.Method=",self.Method)
        print('Time:',t_total) 
        print("p=",self.precision)
        print("r=",self.recall)
        print("f1=",self.f1)
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

    def extra_process_db(self): 
        if self.DataSet == "ACS Income":
            try:
                self.fh_clean.db[:, 9] = self.fh_clean.db[:, 9].astype(float).astype(int).astype(str)
                self.fh_clean.db[:, 11] = self.fh_clean.db[:, 11].astype(float).astype(int).astype(str)
                
            except: 
                print("arr dirty")
                print(self.fh_clean.db[:5, :])


# For Classification
def train_and_eval_mlp_multiclass(
    db_train: np.ndarray,
    db_test: np.ndarray,
    hidden_layer_sizes=(100,),
    max_iter=300,
    random_state=42
):

    X_train = db_train[:, :-1].astype(float)
    y_train_str = db_train[:, -1]

    X_test = db_test[:, :-1].astype(float)
    y_test_str = db_test[:, -1]


    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train_str)
    known_labels = set(label_encoder.classes_)
    mask = np.isin(y_test_str, list(known_labels))

    if not np.all(mask):
        dropped = set(y_test_str[~mask])
        # print(f"[Warning] Unknown labels found in test set, dropped samples: {dropped}")

    X_test = X_test[mask]
    y_test_str = y_test_str[mask]

    if len(y_test_str) == 0:
        raise ValueError("Test set is empty after filtering unknown labels, cannot evaluate model")

    y_test = label_encoder.transform(y_test_str)


    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation="relu",
        solver="adam",
        max_iter=max_iter,
        random_state=random_state
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = round(accuracy_score(y_test, y_pred), 3)

    return acc, model, label_encoder


# For Regression
def train_and_eval_regression(
    db_train: np.ndarray,
    db_test: np.ndarray,
    target: int,
):

    n_train, m = db_train.shape
    n_test, _ = db_test.shape
    
    label_encoders = {}
    db_train_processed = db_train.copy()
    db_test_processed = db_test.copy()
    
    def is_valid(val):
        if val is None:
            return False
        if isinstance(val, float) and np.isnan(val):
            return False
        if isinstance(val, str) and val == '':
            return False
        return True
    
    for col in range(m):
        is_string_col = (db_train[:, col].dtype == object or 
                        db_train[:, col].dtype.kind in ['U', 'S', 'O'])  # Unicode, String, Object
        
        if is_string_col:

            le = LabelEncoder()
        
            train_col = db_train[:, col]
            test_col = db_test[:, col]
            
            all_values = []
            for val in train_col:
                if is_valid(val):
                    all_values.append(str(val))
            for val in test_col:
                if is_valid(val):
                    all_values.append(str(val))
            
            if len(all_values) > 0:
                unique_values = sorted(set(all_values))
                le.fit(unique_values)
                
                value_to_code = {val: code for code, val in enumerate(le.classes_)}
                train_col_encoded = np.zeros(n_train, dtype=int)
                for i, val in enumerate(train_col):
                    if is_valid(val):
                        str_val = str(val)
                        train_col_encoded[i] = value_to_code.get(str_val, 0)
                db_train_processed[:, col] = train_col_encoded
                
                test_col_encoded = np.zeros(n_test, dtype=int)
                for i, val in enumerate(test_col):
                    if is_valid(val):
                        str_val = str(val)
                        test_col_encoded[i] = value_to_code.get(str_val, 0)
                db_test_processed[:, col] = test_col_encoded
                label_encoders[col] = le
    
    try:
        db_train_processed = db_train_processed.astype(float)
        db_test_processed = db_test_processed.astype(float)
    except (ValueError, TypeError):
        for col in range(m):
            try: 
                train_col_float = db_train_processed[:, col].astype(float)
                test_col_float = db_test_processed[:, col].astype(float)
                train_col_float = np.nan_to_num(train_col_float, nan=0.0)
                test_col_float = np.nan_to_num(test_col_float, nan=0.0)
                db_train_processed[:, col] = train_col_float
                db_test_processed[:, col] = test_col_float
            except (ValueError, TypeError):
                db_train_processed[:, col] = 0.0
                db_test_processed[:, col] = 0.0
    print("Finished str2num")
    feature_cols = [i for i in range(m) if i != target]
    
    X_train = db_train_processed[:, feature_cols]
    y_train = db_train_processed[:, target]
    
    X_test = db_test_processed[:, feature_cols]
    y_test = db_test_processed[:, target]
    
    X_train = np.nan_to_num(X_train, nan=0.0)
    y_train = np.nan_to_num(y_train, nan=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0)
    y_test = np.nan_to_num(y_test, nan=0.0)
    print("Finished getting target")
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("Finished training")
    
    y_pred = model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print("Finished testing")
    print("RMSE=",rmse)
    return rmse





