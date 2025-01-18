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
    
    # hard-coded values, where to look for files
    keywords_file = "../data/neural_network_y.csv"
    experiments_folder = "../local/run_pto_experiments_server_ns"
    output_file = os.path.join(experiments_folder, "all_tasks_summary.csv")
    
    # it's probably better to use acronyms in the table, otherwise we get
    # column names that are hard to read
    technique2acronym = {
     'random_search' : 'RS',
     'hill_climber' : 'HC',
     'genetic_algorithm' : 'GA',
     'particle_swarm_optimisation' : 'PSO',
     'novelty_search' : 'NS'
     }
    
    experiment_files = [os.path.join(experiments_folder, f) 
                        for f in os.listdir(experiments_folder)
                        if f.startswith("task")]
    
    print("Found a total of %d experiment files: %s" %
          (len(experiment_files), str(experiment_files)))
    
    results_dictionary = {
        'task_id' : [],
        'n_keywords' : [],
        }
    
    # also, load the file with the keywords per experiment
    # and compute the total number of keywords per row (corresponding to task)
    df_keywords = pd.read_csv(keywords_file)
    numeric_cols = df_keywords.select_dtypes(include="number")
    df_keywords["row_sum"] = numeric_cols.sum(axis=1)
    
    for experiment_file in experiment_files :
        
        # task results file names are expected to be like "task_3c9b0459_results.csv"
        task_id = os.path.basename(experiment_file).split("_")[1]
        results_dictionary['task_id'].append(task_id)
        
        # another thing to keep track of is the total number of keywords in task
        n_keywords = df_keywords[df_keywords["task_id"] == task_id]["row_sum"].values[0]
        results_dictionary['n_keywords'].append(n_keywords)
        
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
                
                key_name = technique2acronym[technique_name] + "_" + c + "_mean"
                if key_name not in results_dictionary :
                    results_dictionary[key_name] = []
                    
                results_dictionary[key_name].append("%.2f (%.2f)" % (mean, stdev))
                
                # let's also compute the median and the deviation around the median
                median = np.median(df_t[c].values)
                mad = np.median(np.absolute(df_t[c].values - median))
                
                key_name = technique2acronym[technique_name] + "_" + c + "_median"
                if key_name not in results_dictionary :
                    results_dictionary[key_name] = []
                
                results_dictionary[key_name].append("%.2f (%.2f)" % (median, mad))
        
        # there is one more statistic we are interested in: the number of times 
        # the training or test fitness is not zero
        key_base = "not_global_optimum"
        for technique_name in technique_names :
            df_t = df[df["technique"] == technique_name]
            n_repetitions = df_t["test_fitness"].values.shape[0]
            n_not_optimum = sum(df_t["test_fitness"].values != 0)
            
            key_name = technique2acronym[technique_name] + "_" + key_base
            if key_name not in results_dictionary :
                results_dictionary[key_name] = []
                
            results_dictionary[key_name].append("\"%d/%d\"" %
                                                (n_not_optimum, n_repetitions))
            
        key_base = "not_zero_training"
        for technique_name in technique_names :
            df_t = df[df["technique"] == technique_name]
            n_repetitions = df_t["train_fitness"].values.shape[0]
            n_not_zero = sum(df_t["train_fitness"].values != 0)
            
            key_name = technique2acronym[technique_name] + "_" + key_base
            if key_name not in results_dictionary :
                results_dictionary[key_name] = []
                
            results_dictionary[key_name].append("\"%d/%d\"" %
                                                (n_not_zero, n_repetitions))
    
    # there is a possibility that not all results for all techniques are available,
    # so here we are making a check and removing all columns for which we do
    # not have all results
    maximum_length = max([len(results_dictionary[k]) for k in results_dictionary])
    keys_to_be_removed = [k for k in results_dictionary 
                          if len(results_dictionary[k]) < maximum_length]
    for k in keys_to_be_removed :
        results_dictionary.pop(k, None)
    
    if len(keys_to_be_removed) > 0 :
        print("Some of the columns do not have enough results, and will be removed:",
              keys_to_be_removed)
        
    df_output = pd.DataFrame.from_dict(results_dictionary)
    df_output.to_csv(output_file, index=False)