# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 14:36:04 2024

Script to check the correctness of programs in functional form.

@author: Alberto
"""
import os
import sys

# local library
import common

if __name__ == "__main__" :
    
    # hard-coded values
    functional_form_file = "../old_unorganized_src/functional_dsl_functions.py"
    arc_tasks_folder = "../data/re_arc/tasks"
    
    # now, there are several ways of dealing with this file, as it is basically
    # a series of macros with different variable names. A first try: I import it
    # and then get the variable names to iterate over
    sys.path.append("../old_unorganized_src/")
    from functional_dsl_functions import *
    
    # get the names and content of all (global) variables that start with 'verify',
    # create a list with ['task_id', 'function_code'] elements
    functional_functions_names = [[v.split("_")[1], c] for v, c in globals().items() if v.startswith("verify")]
    
    # before starting the serious stuff, we also need to import all stuff from
    # Hodel's DSL
    from dsl import *
    
    # data structures to keep track of problematic tasks
    problematic_tasks = dict()
    
    # now we have a dictionary with function name -> string containing its code
    # but python has ways to run code inside a string, using exec(); still, we
    # need to perform some magic with the local variables, to avoid issues
    for task_id, function_code in functional_functions_names :
        
        # add code to actually make the function function a real function
        function_code = "O_predicted = " + function_code.strip()
        
        # now, we need to load the input grid(s) corresponding to the task,
        # load all the necessary data
        pairs = common.load_arc_task_from_json(
            os.path.join(arc_tasks_folder, task_id + ".json"), output="tuple")
        n_pairs = len(pairs["input"])
        print("For task %s, found a total of %d input-output pairs!" %
              (task_id, n_pairs))
        
        n_errors = 0
        for index in range(0, n_pairs) :
            print("- Now evaluating input/output pair %d/%d..." %
                  (index+1, n_pairs), end = '')
            
            I = pairs["input"][index]
            O_measured = pairs["output"][index]
            
            try :
                local_variables = locals().copy()
                exec(function_code, globals(), local_variables)
                O_predicted = local_variables["O_predicted"]
                
                # this might be marked as an error by your IDE, but it's not;
                # O_predicted is added to the local variables by exec()
                #print(O_predicted)
                #print(O_measured)
                
                # one of the two grids is a series of tuples (...), the other
                # one is a numpy array, but I can just perform a pixel-by-pixel comparison
                grids_identical = common.are_grids_pixel_identical(O_measured, O_predicted)
                if grids_identical :
                    print("ok!")
                else :
                    print("ERROR!")
                    n_errors += 1
            
            except Exception as e :
                print("EXCEPTION!")
                n_errors += 1
            
        if n_errors != 0 :
            print("The functional verifier for task \"%s\" generated %d errors!" %
                  (task_id, n_errors))
            problematic_tasks[task_id] = n_errors
            
    n_verifiers = len(functional_functions_names)
    n_problematic = len(problematic_tasks)
    print("Final report: %d/%d verifiers created issues" % (n_problematic, n_verifiers))
    for task_id, n_errors in problematic_tasks.items() :
        print("- task \"%s\": %d errors" % (task_id, n_errors))
        