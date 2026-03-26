from  gurobipy import Model, GRB, LinExpr
from util.Assist import *
from copy import deepcopy
from random import randint
from time import time,perf_counter
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing as mp

class ILP_LP():
    def __init__(self, Ic_obj, Lij,Pair={},k_L=0,if_KeepIc_obj=True,Cf_Set=[],max_workers=None,g_rate=None):
        if if_KeepIc_obj:
            self.Ic_obj=Ic_obj
            self.Ic_obj_copy=deepcopy(Ic_obj)
            self.Ic=sorted(Ic_obj.keys())
        else: 
            self.Cf_Set=Cf_Set 
            self.Ic=sorted(Cf_Set.keys())
        self.Lij=Lij
        self.Pair=Pair
        self.n=len(Lij)
        self.k_L=k_L
        self.halfX=[-1]
        self.IN=[]
        self.CliqueSet=set()
        self.TrueCliqueSet=set()
        self.m=0
        self.resX=None 
        self.resY=None
        self.max_workers=max_workers
        self.g_rate=g_rate



    def LP_Solver(self,if_pos=False,if_binary=False,CliqueSet=set(),if_CheckMem=False):     
        if if_pos:
            self.Pos()
        if self.m==0:
            self.m=Model()
            self.m.ModelSense=GRB.MAXIMIZE
            self.VarDict_x={}
            self.VarDict_y={}

            for i in self.Ic:
                if if_binary:
                    self.VarDict_x[i]=self.m.addVar(lb=0,vtype=GRB.BINARY,name=f'x_{i}')
                else:
                    self.VarDict_x[i]=self.m.addVar(lb=0,vtype=GRB.CONTINUOUS,name=f'x_{i}')
            t1=perf_counter()


            Ic_set = set(self.Ic)
            topk_sets = {}  
            for i in range(self.n):
                if self.k_L > 0 and len(self.Lij[i]) > 0:
                    topk_sets[i] = set(list(self.Lij[i].keys())[:self.k_L])
                else:
                    topk_sets[i] = set()

            num_vars=len(self.VarDict_x)
            num_vars_original=len(self.VarDict_x)
            judge1_cache = {}  
            for i in range(self.n):
                judge1_cache[i] = i in Ic_set
                Li_temp = topk_sets[i]
                
                for j in self.Lij[i].keys():
                    if i == j:
                        continue
                    judge1 = judge1_cache[i]
                    judge2 = j in Ic_set
                    judge3 = j in Li_temp
                    if not judge1 and not judge2 and judge3:
                        self.VarDict_y[(i,j)]=1
                    elif judge1 and not judge2 and judge3:
                        self.VarDict_y[(i,j)]=self.VarDict_x[i]
                    elif not judge1 and judge2 and judge3:
                        self.VarDict_y[(i,j)]=self.VarDict_x[j]
                    elif judge1 and judge2 and j in self.Cf_Set[i]:
                        self.VarDict_y[(i,j)]=0
                    else: 
                        num_vars+=1
                        if if_binary:
                            self.VarDict_y[(i,j)]=self.m.addVar(lb=0,ub=1,vtype=GRB.BINARY,name=f'y_{i}_{j}',obj=self.Lij[i][j])
                        else:
                            self.VarDict_y[(i,j)]=self.m.addVar(lb=0,ub=1,vtype=GRB.CONTINUOUS,name=f'y_{i}_{j}',obj=self.Lij[i][j])
                    num_vars_original+=1
            t2=perf_counter()
            print("t2-t1:",t2-t1)
            print("Original vars=",num_vars_original)
            print("Speed up vars=",num_vars)
            print("Cutting rate=",round((num_vars_original-num_vars)/num_vars_original,3))

            j_to_i_map = {}  
            for i in range(self.n):
                for j in self.Lij[i].keys():
                    if i != j:
                        if j not in j_to_i_map:
                            j_to_i_map[j] = set()
                        j_to_i_map[j].add(i)

            for i in self.Ic:
                if i in self.Cf_Set[i]: 
                    self.m.addConstr(self.VarDict_x[i]==0)
                    continue 
                for j in self.Cf_Set[i]:
                    if j<=i:
                        continue 
                    self.m.addConstr(self.VarDict_x[i]+self.VarDict_x[j]<=1)
            
            for j in self.Ic:
                if j in j_to_i_map:
                    for i in j_to_i_map[j]:
                        y_val = self.VarDict_y[(i,j)]
                        self.m.addConstr(self.VarDict_x[j] - y_val >= 0)
            
            for i in self.Ic:
                for j in self.Lij[i].keys():
                    if i == j:
                        continue
                    y_val = self.VarDict_y[(i,j)]
                    self.m.addConstr(self.VarDict_x[i] - y_val >= 0)
            
            for i in self.Ic:           
                exp=LinExpr()
                for j in self.Lij[i].keys():
                    if i==j:
                        continue
                    y_val = self.VarDict_y[(i,j)]
                    if isinstance(y_val, (int, float)):
                        exp.addConstant(y_val)
                    else:
                        exp.addTerms(1.0, y_val)
                self.m.addConstr(exp-self.k_L*self.VarDict_x[i]<=0)  

            for i in range(self.n):
                if i not in Ic_set:
                    exp=LinExpr()
                    for j in self.Lij[i].keys():
                        if i==j:
                            continue
                        y_val = self.VarDict_y[(i,j)]
                        if isinstance(y_val, (int, float)):
                            exp.addConstant(y_val)
                        else:
                            exp.addTerms(1.0, y_val)
                    self.m.addConstr(exp-self.k_L<=0)  
        
        else: 
            for cq in self.NewCliqueSet:
                exp=LinExpr()
                for i in cq:
                    exp.addTerms(1.0,self.VarDict_x[i])
                self.m.addConstr(exp-1<=0)
        

        if if_CheckMem:
            print('--Gurobi Memory Cost:')
            CalcMem(self.m)
        ta=perf_counter()
        self.m.Params.OutputFlag = 0
        tb=perf_counter()
        print("tb-ta=",tb-ta)
        self.m.optimize()
        if self.m.Status == GRB.INFEASIBLE:
            print("Model is infeasible")
        elif self.m.Status == GRB.UNBOUNDED:
            print("Model is unbounded")
        else:
            print("Model solved successfully")
        self.resX={i:self.VarDict_x[i].X for i in self.VarDict_x.keys()}
        self.resY={i:self.VarDict_y[i].X for i in self.VarDict_y.keys() if not isinstance(self.VarDict_y[i],(float,int))}

        self.IN=[x for x in self.resX.keys() if self.resX[x] < 0.5]     
        self.halfX=[x for x in self.resX.keys() if self.resX[x] == 0.5]
        return self.IN




    def Pos(self):      
        tem=[self.Lij[i][j] for i in self.Lij for j in self.Lij[i]]
        Lm,LM=min(tem),max(tem)
        if self.g_rate is not None:
            LM=LM*self.g_rate
        self.Lij={i:{j: LM-self.Lij[i][j] for j in self.Lij[i].keys()} for i in self.Lij}

    
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
            for j in self.Lij[i].keys():
                self.Lij[i][j]*=self.pct[i]


    def Minimization(self,IN):   
        self.Li={i:sum(find_top_k(self.Lij[i].values(), self.k_L, Type='largest')[0]) for i in self.Lij.keys()}
        PutBack=set()
        sorted_Li=sorted(self.Li, key=lambda k: self.Li[k], reverse=True)
        sorted_Li=[i for i in sorted_Li if i in IN]
        for i in sorted_Li:
            if i in IN:
                judge=1
                for j in self.Cf_Set.keys():
                    if j in IN and j not in PutBack:
                        continue
                    if i in self.Cf_Set.keys():
                        judge=0
                        break
                if judge:
                    PutBack.update({i})
        IN=set(IN)-PutBack
        return IN

    def FindClique(self):    
        CliqueSet=set()
        self.NewCliqueSet=set() 
        self.halfX_copy=self.halfX*1
        while len(self.halfX)>0:
            i=self.halfX[0]   
            temset={i}
            self.halfX.remove(i)
            CS1=[x for x in self.Cf_Set[i] if x in self.halfX_copy]      
            CS2=[x for x in self.Cf_Set[i] if x not in CS1]
            for j in CS1:
                if all(j in self.Cf_Set[x] for x in temset):       
                    temset.add(j)
                    if j in self.halfX:
                        self.halfX.remove(j)
            for j in CS2:
                if all(j in self.Cf_Set[x] for x in temset):
                    temset.add(j)

            if len(temset)>1:
                CliqueSet.add(tuple(temset))  
            if len(temset)>2:
                if tuple(temset) not in self.TrueCliqueSet:
                    self.NewCliqueSet.add(tuple(temset))    
                self.TrueCliqueSet.add(tuple(temset))    
        return CliqueSet


    def Solve_with_Clique(self,if_CheckMem=False,Max_Turn=1e9):
        t1=perf_counter()  
        self.LP_Solver(if_pos=False,if_binary=False,if_CheckMem=if_CheckMem) 
        t2=perf_counter()
        print("Solving time round -- 0:",round(t2-t1,3))
        turns=0
        while len(self.halfX)>0 and turns<Max_Turn:
            t1=perf_counter()
            halfX={x:self.resX[x] for x in self.resX.keys() if self.resX[x] not in {0,1}}  
            self.CliqueSet=self.FindClique()               
            if len(self.NewCliqueSet)==0:
                self.IN=[x for x in self.resX.keys() if self.resX[x]<=0.5]
                return self.IN
            else:
                print('\nGurobi Turn', turns)
                t3=perf_counter()
                self.LP_Solver(if_pos=False,if_binary=False,CliqueSet=self.TrueCliqueSet,if_CheckMem=if_CheckMem)
                t4=perf_counter()

            turns+=1
            t2=perf_counter()
            print(f"Solving time round -- {turns}:",round(t2-t1,3))
        self.IN=[x for x in self.resX.keys() if self.resX[x]<=0.5]
        return self.IN







