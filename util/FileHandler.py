import numpy as np
import pandas as pd
import os
import chardet

class FileHandler():
    def __init__(self,dataName,LineRange='All'):
        self.LineRange=LineRange  
        self.data=pd.DataFrame()  
        self.db=np.array([])       
        self.attr_id={}            
        self.id_attr={}             
        self.dataName=dataName     
        path=os.path.dirname(__file__)
        path=os.path.dirname(path)+'/data/'
        self.path=path


    def Loader(self,filename,sep=',',header=0,index_col=None,if_all_str=False):      # 读数
        tem=self.path+filename
        self.data = pd.read_csv(tem, sep=sep, header=header, index_col=index_col, encoding_errors='ignore')
        if self.LineRange != 'All':
            self.data=self.data.iloc[self.LineRange[0]:self.LineRange[1]+1]
        self.db=self.data.values*1
        if if_all_str:
            self.db.astype(str)

    def AttrId(self):
        i=0
        for c in self.data.columns:
            self.attr_id[c]=i
            self.id_attr[i]=c
            i=i+1


    def Saver(self,data,filename,sep=','):
        tem=self.path+self.dataName+'/result/'
        if not os.path.exists(tem):
            os.makedirs(tem)
        tem=tem+filename
        data.to_csv(tem,sep=sep)


    def fullna(self):
        for i in range(self.db.shape[0]):
            for j in range(self.db.shape[1]):
                if isinstance(self.db[i, j], float) and np.isnan(self.db[i, j]):
                    self.db[i,j]=0
        self.convert_columns_to_str()
        
    def convert_columns_to_str(self, threshold=0.1):
        for j in range(self.db.shape[1]):
            str_count = np.sum([isinstance(item, str) for item in self.db[:, j]])
            str_ratio = str_count / self.db.shape[0]
            if str_ratio > threshold:
                self.db[:, j] = self.db[:, j].astype(str)



