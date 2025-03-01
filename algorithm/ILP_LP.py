from  gurobipy import Model, GRB, LinExpr
from util.Assist import *
from copy import deepcopy
from random import randint

class ILP_LP():
    def __init__(self, Ic_obj, Lij,Pair={},k_L=0,if_KeepIc_obj=True,Cf_Set=[]):
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
                    self.VarDict_x[i]=self.m.addVar(lb=0,vtype=GRB.CONTINUOUS,name=f'x_{i}')   # x_i=1 留，x_i=0删。
            for i in range(self.n):
                for j in self.Lij[i].keys():
                    if i==j:
                        continue
                    if if_binary:
                        self.VarDict_y[(i,j)]=self.m.addVar(lb=0,ub=1,vtype=GRB.BINARY,name=f'y_{i}_{j}',obj=self.Lij[i][j])
                    else:
                        self.VarDict_y[(i,j)]=self.m.addVar(lb=0,ub=1,vtype=GRB.CONTINUOUS,name=f'y_{i}_{j}',obj=self.Lij[i][j])
            for i in self.Ic:
                if i in self.Cf_Set[i]: 
                    self.m.addConstr(self.VarDict_x[i]==0)
                    continue 
                for j in self.Cf_Set[i]:
                    if j<=i:
                        continue 
                    self.m.addConstr(self.VarDict_x[i]+self.VarDict_x[j]<=1)
            for j in self.Ic:           
                for i in range(self.n): 
                    if j not in self.Lij[i]:
                        continue
                    self.m.addConstr(self.VarDict_x[j]-self.VarDict_y[(i,j)]>=0)   
            for i in self.Ic:           
                for j in range(self.n): 
                    if j not in self.Lij[i]:
                        continue
                    self.m.addConstr(self.VarDict_x[i]-self.VarDict_y[(i,j)]>=0)  
            for i in self.Ic:           
                exp=LinExpr()
                for j in self.Lij[i].keys():
                    if i==j:
                        continue
                    exp.addTerms(1.0, self.VarDict_y[(i,j)])
                self.m.addConstr(exp-self.k_L*self.VarDict_x[i]<=0)  

            for i in range(self.n):
                if i not in self.Ic:
                    exp=LinExpr()
                    for j in self.Lij[i].keys():
                        if i==j:
                            continue
                        exp.addTerms(1.0, self.VarDict_y[(i,j)])
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
        self.m.Params.OutputFlag = 0
        self.m.optimize()
        if self.m.Status == GRB.INFEASIBLE:
            print("Model is infeasible")
        elif self.m.Status == GRB.UNBOUNDED:
            print("Model is unbounded")
        else:
            print("Model solved successfully")
        self.resX={i:self.VarDict_x[i].X for i in self.VarDict_x.keys()}
        self.IN=[x for x in self.resX.keys() if self.resX[x] < 0.5]     
        self.halfX=[x for x in self.resX.keys() if self.resX[x] == 0.5]
        return self.IN

    def Pos(self):      
        tem=[self.Lij[i][j] for i in self.Lij for j in self.Lij[i]]
        Lm,LM=min(tem),max(tem)
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
        self.LP_Solver(if_pos=False,if_binary=False,if_CheckMem=if_CheckMem)   
        turns=0
        while len(self.halfX)>0 and turns<Max_Turn:
            halfX={x:self.resX[x] for x in self.resX.keys() if self.resX[x] not in {0,1}}  
            self.CliqueSet=self.FindClique()               
            if len(self.NewCliqueSet)==0:
                self.IN=[x for x in self.resX.keys() if self.resX[x]<=0.5]
                return self.IN
            else:
                print('Gurobi Turn', turns)
                self.LP_Solver(if_pos=False,if_binary=False,CliqueSet=self.TrueCliqueSet,if_CheckMem=if_CheckMem)       
            turns+=1
        self.IN=[x for x in self.resX.keys() if self.resX[x]<=0.5]
        return self.IN

















# # 重要参数：
# half_in:上一轮整数，这一轮分数
# coverdX：half_in中，已经被覆盖的
# self.NewCliqueSet：每一轮新找到的团
# self.TrueCliqueSet：累积下来的团
# self.TrueCliqueSet2：每一轮新团（id=轮数）
# self.TrueCliqueSet3：每一轮新团（id=轮数），还有构成它的元组
# self.CliqueSet:包含大小等于2的团 （除了这个都是3以上的团）
# self.resXelse：不为0，1的元组
# tight_constraint：团中有大于0.5的元组
# untight_constraint：团中没有大于0.5的元组
# rhcq：极小层面上的冗余元组
# t_num：每个元组的取值






