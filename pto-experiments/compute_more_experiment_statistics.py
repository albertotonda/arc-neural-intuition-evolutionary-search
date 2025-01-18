# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 19:39:46 2024

@author: Alberto
"""
import numpy as np
import pandas as pd

if __name__ == "__main__" :
    
    csv_file = "../local/run_pto_experiments_server_ns/all_tasks_summary.csv"
    
    df = pd.read_csv(csv_file)
    
    # who takes less iterations to reach a solution?
    columns_median = [c for c in df.columns if c.endswith("_generations_median")]
    
    for index, row in df.iterrows() :
        
        # create a list with technique name and median value
        performances = []
        for c in columns_median :
            technique = c.split("_")[0]
            value = float(row[c].split(" ")[0])
            
            if technique != 'RS' :
                value = value * 100.0
            
            performances.append([technique, value])
            
        # sort the list
        performances = sorted(performances, key=lambda x : x[1])
        
        # print out some stuff
        task_id = row['task_id']
        print("Task \"%s\": %s" % (task_id, str([p[0] for p in performances])))