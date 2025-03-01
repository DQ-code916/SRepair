import numpy as np


class ResultAnalysis():
    def __init__(self,db_clean,db_dirty,n,m):
        self.db_clean=db_clean
        self.db_dirty=db_dirty
        self.db_repaired=np.array([])
        self.n,self.m=n,m
        self.TrueError=set()
        self.Changed=set()   

    def S_Repair_GroundTruth(self,data=[]):
        for i in range(self.n):
            for j in range(self.m):
                tem1,tem2=self.db_clean[i,j],self.db_dirty[i,j]
                if self.db_clean[i,j]!=self.db_dirty[i,j]:
                    self.TrueError.update({i})
                    break
        
    
    def S_Repair_Changed(self,IN):     
        self.Changed=IN
                    
    def S_Repair_Calc_3_metric(self):
        if not isinstance(self.Changed,set):
            self.Changed=set(self.Changed)
        if not isinstance(self.TrueError,set):
            self.TrueError=set(self.TrueError)
        CorrectChange=self.Changed & self.TrueError
        if len(self.Changed)!=0:
            precision=len(CorrectChange)/len(self.Changed)
        else: 
            precision=0
        recall=len(CorrectChange)/len(self.TrueError)
        if (precision+recall)!=0:
            f1=2*precision*recall/(precision+recall)
        else: 
            f1=0

        return round(precision,3),round(recall,3),round(f1,3)
    
    def SetDbRepair(self,db_repair):
        self.db_repaired=db_repair

    def WrongChange(self):
        correct_change=self.TrueError & self.Changed
        wrong_change=self.Changed-correct_change
        ignore_error=self.TrueError-correct_change
        return correct_change,wrong_change,ignore_error
