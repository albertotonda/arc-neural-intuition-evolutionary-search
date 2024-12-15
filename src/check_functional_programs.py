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
    
    tasks_to_be_checked = ['0520fde7',
     '05269061',
     '0a938d79',
     '0e206a2e',
     '150deff5',
     '1b2d62fb',
     '1b60fb0c',
     '2013d3e2',
     '2204b7a8',
     '234bbc79',
     '23581191',
     '272f95fa',
     '27a28665',
     '28e73c20',
     '2bcee788',
     '2dd70a9a',
     '3428a4f5',
     '3618c87e',
     '3631a71a',
     '36d67576',
     '36fdfd69',
     '3906de3d',
     '39e1d7f9',
     '3ac3eb23',
     '3bd67248',
     '3e980e27',
     '3eda0437',
     '4093f84a',
     '496994bd',
     '4be741c5',
     '50846271',
     '508bd3b6',
     '539a4f51',
     '54d82841',
     '5521c0d9',
     '56dc2b01',
     '6430c8c4',
     '662c240a',
     '673ef223',
     '6855a6e4',
     '694f12f3',
     '6a1e5592',
     '6d58a25d',
     '73251a56',
     '7447852a',
     '746b3537',
     '75b8110e',
     '760b3cac',
     '77fdfe62',
     '7b7f7511',
     '82819916',
     '83302e8f',
     '846bdb03',
     '855e0971',
     '8731374e',
     '8e1813be',
     '8e5a5113',
     '90c28cc7',
     '93b581b8',
     '94f9d214',
     '952a094c',
     '995c5fa3',
     '99b1bc43',
     '9d9215db',
     '9dfd6313',
     'a3df8b1e',
     'a64e4611',
     'a65b410d',
     'a68b268e',
     'a78176bb',
     'a8c38be5',
     'af902bf9',
     'b0c4d837',
     'b7249182',
     'b8cdaf2b',
     'bbc9ae5d',
     'bc1d5164',
     'bd4472b8',
     'beb8660c',
     'c9f8e694',
     'caa06a1f',
     'ce4f8723',
     'd4469b4b',
     'd8c310e9',
     'd9f24cd1',
     'dae9d2b5',
     'db3e9e38',
     'ddf7fa4f',
     'de1cd16c',
     'e179c5f4',
     'e26a3af2',
     'e6721834',
     'e98196ab',
     'ea786f4a',
     'eb5a1d5d',
     'f15e1fac',
     'f25ffba3',
     'f2829549',
     'f8a8fe49',
     'fafffa47',
     'ff805c23']
    
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
        
        n_errors = 0
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
                    n_errors = n_errors # horrible line
                else :
                    #print("ERROR!")
                    n_errors += 1
            
            except Exception as e :
                #print("EXCEPTION!")
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
        