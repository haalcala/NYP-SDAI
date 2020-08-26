# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 19:59:36 2020

@author: Harold
"""

import pandas as pd
from datetime import datetime

class DSUtil:
    def load_csv(self, csv, **args):
        print("args:", args)

        df = pd.read_csv(csv, **args)
        self.df = df
        
        print(df.info())
        

    def blow_my_mind(self):
        df = self.df
        
        for col in df.columns:
            print(f"unique {df[{col}].nunique()}") 
        
        # print column unique values and counts
        
        print(f"---------------------------------------------------------------- Has any missing (Nan) data? {df.isnull().values.any()}")
        
        if df.isnull().values.any():
            for col in df.columns:
                print(f"---------------------------------------------------------------- Column {col} has missing data? {df[col].isnull().values.any()}")
                
        print("----------------------------------------------------------------")
        
        for col in df.columns:
            print("value count:", df[col].value_counts())
            print("----------------------------------------------------")
            
        # TODO: detect missing values and warn
            
    def drop_columns(self, columns):    
        df = self.df
        
        for col in columns:
            try:
                df = df.drop(col, axis=1)
            except:
                pass
            
        self.df = df
        
    def activity_wrapper(self, activity_name, activity_fn):
        print(activity_name,"...")
        start = datetime.now()
        activity_fn()
        elapsed = datetime.now() - start
        print(activity_name,"... done.", elapsed.total_seconds(),"seconds")
        return elapsed
        
    def get_dummies(self):
        return pd.get_dummies(self.df)



if __name__ == "__main__":
    pass

def test():
    print("Tested!!!")