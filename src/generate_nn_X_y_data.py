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

import numpy as np
import pandas as pd
import re as regex

def extract_keywords_from_dsl(file_name : str) -> list[str] :
    
    # list of keywords that will be returned
    keywords = {'functions' : [], 'constants' : []}
    
    # load file
    text = None
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

def extract_class_labels_from_verifiers(verifiers_file : str, 
                                        operators : list[str], terminals : list[str],
                                        keyword_to_index : dict,
                                        ) -> dict :
    
    # 'class_labels' is a matrix with: task_id, and then a value 0/1 for each
    # class_name which was found in the verifier solving that particular task
    class_labels = {"task_id" : []}
    
    # now, the matrix has size (n_tasks, n_classes); but n_tasks is unknown
    # at the beginning, so we are going to start from scratch and just stack numpy
    # arrays on top of each other
    Y = None
    n_classes = len(keyword_to_index)
    
    # prepare another data structure that will be useful later
    sorted_class_names = [""] * n_classes
    for class_name, class_index in keyword_to_index.items() :
        sorted_class_names[class_index] = class_name
    
    # read the file and start parsing it
    verifiers_text = None
    with open(verifiers_file, "r") as fp :
        verifiers_text = fp.readlines()
    
    line_index = 0
    while line_index < len(verifiers_text) :
        line = verifiers_text[line_index]
        
        # regular expression that matches the start of a verifier function
        m = regex.search("verify\_([\w]+)", line)
        if m is not None :
            # we are inside a function, so let's store the task id
            task_id = m.group(1)
            class_labels["task_id"].append(task_id)
            print("Found verifier for task \"%s\"!" % task_id)
            
            # all the function is assumed to be stored in a single row
            line = verifiers_text[line_index+1]
            
            # allocate numpy vector
            y_task = np.zeros((1, n_classes))
            
            # this is for debugging
            task_keywords_found = []
            
            # naively checking for the presence of certain keywords can
            # work for most functions, but not for everything; we need to
            # create separate regular expressions for each operator and terminal
            for operator in operators :
                regular_expression = operator + "[\(|,|\)]"
                m = regex.search(regular_expression, line)
                if m is not None : 
                    operator_index = keyword_to_index[operator]
                    y_task[0,operator_index] = 1
                    task_keywords_found.append(operator)
                    
            for terminal in terminals :
                regular_expression = "[,\s+|\(]" + terminal + "[,|\)]"
                m = regex.search(regular_expression, line)
                if m is not None :
                    terminal_index = keyword_to_index[terminal]
                    y_task[0,terminal_index] = 1
                    task_keywords_found.append(terminal)
            
            print("For task \"%s\", found keywords: %s" % 
                 (task_id, str(task_keywords_found)))
            
            # it's time to stack the current class label vector at the bottom
            # of all the previous stuff
            if Y is None :
                Y = y_task
            else :
                Y = np.concatenate((Y, y_task), axis=0)
        
        line_index += 1
    
    print("Final shape of Y:", Y.shape)
    
    # at this point, take X and place it in the dictionary
    for i, class_name in enumerate(sorted_class_names) :
        class_labels[class_name] = Y[:,i]
    
    return class_labels

if __name__ == "__main__" :
    
    # source file for the DSL
    dsl_file = "../old_unorganized_src/dsl.py"
    # source file for the verifiers
    verifiers_file = "../old_unorganized_src/functional_dsl_functions.py"
    
    # ouput file for the class labels
    class_names_file = "../data/neural_network_class_labels.csv"
    # output file for the class labels
    class_labels_file = "../data/neural_network_y.csv"
    
    # parse dsl_file and get keywords
    print("Parsing keywords from file \"%s\"..." % dsl_file)
    keywords = extract_keywords_from_dsl(dsl_file)
    print("Found a total of %d functions and %d constants" % 
          (len(keywords["functions"]), len(keywords["constants"])))
    
    # save keywords to a file, with a class number associated; "I" and "call"
    # are special, and added separately. "I" is the variable associated to the input grid;
    # "call" is a special function designed to call the function stored in a variable
    all_keywords = ["I"] + sorted(keywords["constants"]) + ["call"] + sorted(keywords["functions"])
    dict_class_names = dict()
    dict_class_names["class_name"] = all_keywords
    dict_class_names["class_label"] = [i for i in range(0, len(dict_class_names["class_name"]))]
    print("Saving results to file \"%s\"..." % class_names_file)
    df_class_names = pd.DataFrame.from_dict(dict_class_names)
    df_class_names.to_csv(class_names_file, index=False)
    
    # now, we can use the information we just extracted to parse the verifiers
    # written by Hodel and get all information out; in order to treat different parts
    # differently during parsing, let's create separate data structures
    operators = ["call"] + sorted(keywords["functions"])
    terminals = ["I"] + sorted(keywords["constants"])
    keyword_to_index = {
        dict_class_names["class_name"][i] : dict_class_names["class_label"][i]
                        for i in range(0, len(dict_class_names["class_name"]))
                        }
    
    dict_class_labels = extract_class_labels_from_verifiers(
        verifiers_file, operators, terminals, keyword_to_index
        )
    print("Found verifiers for a total of %d tasks!" % len(dict_class_labels["task_id"]))
    
    # save everything to a Pandas dataframe
    print("Saving results to file \"%s\"..." % class_labels_file)
    df = pd.DataFrame.from_dict(dict_class_labels)
    df.to_csv(class_labels_file, index=False)