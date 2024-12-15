# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 17:08:10 2024

Further refinement of the verifier analysis: we just want to check if they
have keywords in common

@author: Alberto
"""
import re as regex
import sys

if __name__ == "__main__" :
    functional_form_file = "../old_unorganized_src/functional_dsl_functions.py"
    arc_tasks_folder = "../data/re_arc/tasks"
    
    tasks_to_be_checked = ['2204b7a8',
     '272f95fa',
     '2dd70a9a',
     '73251a56',
     '7447852a',
     '77fdfe62',
     '83302e8f',
     '8e5a5113',
     '93b581b8',
     '952a094c',
     '995c5fa3',
     'a3df8b1e',
     'a78176bb',
     'a8c38be5',
     'bbc9ae5d',
     'caa06a1f',
     'd8c310e9',
     'e179c5f4',
     'ea786f4a']
    
    #tasks_to_be_checked = ['2204b7a8', '272f95fa', '73251a56', '7447852a', 
    #                       '77fdfe62', '83302e8f', '8e5a5113', '952a094c', 
    #                       'a3df8b1e', 'a8c38be5', 'bbc9ae5d', 'caa06a1f', 
    #                       'd8c310e9', 'e179c5f4', 'ea786f4a']
    #tasks_to_be_checked = ['2dd70a9a', '93b581b8', '995c5fa3', 'a78176bb']

    sys.path.append("../old_unorganized_src/")
    from functional_dsl_functions import *
    
    # get the names and content of all (global) variables that start with 'verify',
    # create a list with ['task_id', 'function_code'] elements
    functional_functions_names = [[v.split("_")[1], c] for v, c in globals().items() if v.startswith("verify")]
    
    if len(tasks_to_be_checked) > 0 :
        functional_functions_names = [f for f in functional_functions_names 
                                      if f[0] in tasks_to_be_checked]
        
    # get keywords from each verifier
    task_keywords = {}
    for task_id, code in functional_functions_names :
        
        keywords = set(sorted([s.strip() for s in regex.split("\(|,|\)", code.strip()) if s != ""]))
        print("%s: %s" % (task_id, str(keywords)))
        
        task_keywords[task_id] = keywords
        
    # do they have at least a word in common?
    common_set = None
    for task_id, keywords in task_keywords.items() :
        if common_set is None :
            common_set = keywords
        else :
            common_set = common_set.intersection(keywords)
            
    print("Keywords common to all sets: %s" % str(common_set))
