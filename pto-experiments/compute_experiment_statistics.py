# -*- coding: utf-8 -*-
"""
Created on Sat Dec 21 15:34:35 2024

Basic script to collect some statistics on the experiments

@author: Alberto
"""
import numpy as np
import os
import pandas as pd

if __name__ == "__main__" :
    
    experiments_folder = "../local/run_pto_experiments_server"
    output_file = os.path.join(experiments_folder, "all_tasks_summary.csv")
    
    experiment_files = [os.path.join(experiments_folder, f) 
                        for f in os.listdir(experiments_folder)
                        if f.startswith("task")]
    
    print("Found a total of %d experiment files: %s" %
          (len(experiment_files), str(experiment_files)))
    
    results_dictionary = {
        'task_id' : [],
        }
    
    for experiment_file in experiment_files :
        
        # task results file names are expected to be like "task_3c9b0459_results.csv"
        task_id = os.path.basename(experiment_file).split("_")[1]
        results_dictionary['task_id'].append(task_id)
        
        # load file as DataFrame
        df = pd.read_csv(experiment_file)
        
        # irrelevant columns for statistics
        irrelevant_columns = ["technique", "repetition", "random_seed", "phenotype"]
        relevant_columns = [c for c in df.columns if c not in irrelevant_columns]
        
        # check all techniques in file
        technique_names = df["technique"].unique()

        # compute statistics on columns
        for c in relevant_columns :        
            for technique_name in technique_names :
                # select all row for that technique
                df_t = df[df["technique"] == technique_name]
            
                mean = np.mean(df_t[c].values)
                stdev = np.std(df_t[c].values)
                
                print("Technique \"%s\", %s: mean=%.4f, std=%.4f" %
                      (technique_name, c, mean, stdev))
                
                key_name = technique_name + "_" + c
                if key_name not in results_dictionary :
                    results_dictionary[key_name] = []
                    
                results_dictionary[key_name].append("%.2f (%.2f)" % (mean, stdev))
        
        # there is one more statistic we are interested in: the number of times 
        # the training or test fitness is not zero
        key_base = "not_global_optimum"
        for technique_name in technique_names :
            df_t = df[df["technique"] == technique_name]
            n_repetitions = df_t["test_fitness"].values.shape[0]
            n_not_optimum = sum(df_t["test_fitness"].values != 0)
            
            key_name = technique_name + "_" + key_base
            if key_name not in results_dictionary :
                results_dictionary[key_name] = []
                
            results_dictionary[key_name].append("\"%d/%d\"" %
                                                (n_not_optimum, n_repetitions))
    
    print(results_dictionary)
    df_output = pd.DataFrame.from_dict(results_dictionary)
    df_output.to_csv(output_file, index=False)