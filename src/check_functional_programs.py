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
    
    # now, there are several ways of dealing with this file, as it is basically
    # a series of macros with different variable names. A first try: I import it
    # and then get the variable names to iterate over
    sys.path.append("../old_unorganized_src/")
    from functional_dsl_functions import *
    
    # get the names and content of all (global) variables that start with 'verify',
    # create a list with ['task_id', 'function_code'] elements
    functional_functions_names = [[v.split("_")[1], c] for v, c in globals().items() if v.startswith("verify")]
    
    if len(tasks_to_be_checked) > 0 :
        functional_functions_names = [f for f in functional_functions_names 
                                      if f[0] in tasks_to_be_checked]
    
    n_tasks = len(functional_functions_names)
    print("A total of %d functional verifiers will be checked" % n_tasks)
    
    # before starting the serious stuff, we also need to import all stuff from
    # Hodel's DSL
    from dsl import *
    
    # we also need to import a specialized 'call' function
    from common import call
    
    # data structures to keep track of problematic tasks
    problematic_tasks = dict()
    
    # now we have a dictionary with function name -> string containing its code
    # but python has ways to run code inside a string, using exec(); still, we
    # need to perform some magic with the local variables, to avoid issues
    for task_index, (task_id, function_code) in enumerate(functional_functions_names) :
        
        print("Now analyzing functional verifier for task \"%s\" (%d/%d)..." 
              % (task_id, task_index+1, n_tasks))
        
        # add code to actually make the function function a real function
        function_code = "O_predicted = " + function_code.strip()
        
        # now, we need to load the input grid(s) corresponding to the task,
        # load all the necessary data
        pairs = common.load_arc_task_from_json(
            os.path.join(arc_tasks_folder, task_id + ".json"), output="tuple")
        n_pairs = len(pairs["input"])
        print("For task %s, found a total of %d input-output pairs!" %
              (task_id, n_pairs))
        
        n_pairs = 10 # TODO comment this line, it's just for debugging
        
        errors = []
        for index in range(0, n_pairs) :
            #print("- Now evaluating input/output pair %d/%d..." %
            #      (index+1, n_pairs), end = '')
            
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
                    #print("ok!")
                    pass
                else :
                    #print("ERROR!")
                    errors.append("Input-output grids not matching!")
            
            except Exception as e :
                #print("EXCEPTION!")
                errors.append("Exception: \"%s\"" % str(e))
            
        n_errors = len(errors)
        if n_errors != 0 :
            print("The functional verifier for task \"%s\" generated %d errors: %s" %
                  (task_id, n_errors, str(errors)))
            problematic_tasks[task_id] = errors
            
        # we also want to check the regular verifiers
        print("Now checking regular verifier for the same task...")
        import linear_verifiers
        verifier_name = "verify_" + task_id
        verifier_function = getattr(linear_verifiers, verifier_name)
        linear_verifier_errors = 0
        
        for index in range(0, n_pairs) :
            I = pairs["input"][index]
            O_measured = pairs["output"][index]
            
            O_predicted = verifier_function(I)
            if common.are_grids_pixel_identical(O_measured, O_predicted) == False :
                linear_verifier_errors += 1
                
        print("The linear verifier made %d errors" % linear_verifier_errors)
            
    n_verifiers = len(functional_functions_names)
    n_problematic = len(problematic_tasks)
    print("Final report: %d/%d verifiers created issues" % (n_problematic, n_verifiers))
    for task_id, errors in problematic_tasks.items() :
        print("- task \"%s\": %d errors" % (task_id, len(errors)))
        
    # group them by error type; big assumption, all errors are the same for the same task
    tasks_by_error = {}
    for task_id, errors in problematic_tasks.items() :
        if errors[0] not in tasks_by_error :
            tasks_by_error[errors[0]] = []
        tasks_by_error[errors[0]].append(task_id)
        
    print(tasks_by_error)
    
    
    
        