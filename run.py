from main_core import *
from copy import deepcopy
import math
import numpy as np
import pandas as pd
import os
import pickle
from util.Assist import *
from util.u_repair_tools import *
import matplotlib.pyplot as plt
import random



if __name__ == '__main__':


    DataSet_Set=["flights"]
    MethodSet=[ "Probabilistic","Clique" ]   
    LineRange="All"
      
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
