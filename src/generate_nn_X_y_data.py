# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 15:44:54 2024

Script to extract training information for the neural network "intuition" part
of the framework.

1. using regexps, parse the solver programs written by Michael Hodel (or their 
version in functional form), identify the keywords, corresponding to functions
in Michael Hodel's DSL. These are going to be the classes/labels for the multi-
label classification task.

2. from the corresponding ARC-AGI tasks, find the input-output pairs and convert
them to a pair of 30x30 (padded) grids. This is going to be our X. Save everything
to a dictionary file, to be more easily used later...?

@author: Alberto
"""

import pandas as pd
import re as regex

def extract_keywords_from_dsl(file_name : str) -> list[str] :
    
    # list of keywords that will be returned
    keywords = {'functions' : [], 'constants' : []}
    
    # load file
    text = []
    with open(file_name, "r") as fp :
        text = fp.readlines()
        
    line_index = 0
    while line_index < len(text) :
        
        line = text[line_index]
        
        # check if line matches a function
        m = regex.search("def ([0-9|a-z]+)\(", line)
        if m is not None :
            function_name = m.group(1)
            print("Found a function named \"%s\"!" % function_name)
            
            keywords["functions"].append(function_name)
            
        # the other possibility is that the line matches a constant;
        # constants are all caps, and not indented
        m = regex.search("([A-Z|\_]+)\s+=", line)
        if m is not None :
            constant_name = m.group(1)
            print("Found a constant named \"%s\"!" % constant_name)
            
            keywords["constants"].append(constant_name)
            
        line_index += 1
    
    return keywords
    

if __name__ == "__main__" :
    
    # source file for the DSL
    dsl_file = "../old_unorganized_src/dsl.py"
    # source file for the solver
    
    # ouput file for the class labels
    class_labels_file = "../data/class_labels.csv"
    
    # parse dsl_file and get keywords
    keywords = extract_keywords_from_dsl(dsl_file)
    print("Found a total of %d functions and %d constants" % 
          (len(keywords["functions"]), len(keywords["constants"])))
    
    # save keywords to a file, with a class number associated
    dict_class_labels = dict()
    dict_class_labels["class_name"] = ["I"] + sorted(keywords["constants"]) + sorted(keywords["functions"])
    dict_class_labels["class_label"] = [i for i in range(0, len(dict_class_labels["class_name"]))]
    df_class_labels = pd.DataFrame.from_dict(dict_class_labels)
    df_class_labels.to_csv(class_labels_file, index=False)