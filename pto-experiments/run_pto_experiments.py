# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 17:13:50 2024

Run Program Trace Optimization experiments on several tasks, comparing different
techniques: Random Search, Hill Climber, Genetic Algorithms, Particle Swarm Optimization.

@author: Alberto Tonda
"""
import json
import numpy as np
import os
import pandas as pd
import random
import time

# imports from the Program Trace Optimization module
from pto import run, rnd

# we need a few imports from Michael Hodel's Domain Specific Language (DSL)
import sys
sys.path.append("../local/re-arc")
from dsl import *
import utils

# local scripts
from common import initialize_logging, call, close_logging

# functions
def generate_random_expression(node_tree, max_depth=10):
    """
    Generate random DSL expressions while avoiding specified keywords/functions
    
    Args:
        node_tree: The DSL node tree structure
        max_depth: Maximum depth of generated expression
        
    Returns:
        A randomly generated expression string
    """
    def expand_node(node, current_depth = 0):
        # Stop if max depth reached or node not in tree
        if node not in node_tree:
            return node
        
        if current_depth >= max_depth:
            return node
        
        # Select a random child list for this node
        children_lists = node_tree[node]
        if not children_lists:
            return node
        children = rnd.choice(list(map(tuple,children_lists)))
        
        # Recursively expand children
        expanded_children = [expand_node(child, current_depth + 1).lstrip('_') for child in children]
        
        # Construct function call
        return f"{node}({', '.join(expanded_children)})"
    
    # Start from root node
    root_lists = node_tree.get('root', [[]])
    
    # Choose random valid root
    root = rnd.choice(list(map(tuple,root_lists)))[0]
    
    return expand_node(root)

def grid_distance(grid1, grid2):
    """
    Calculate the distance between two grids by counting mismatches after padding.
    
    Args:
        grid1: First grid (2D list or numpy array)
        grid2: Second grid (2D list or numpy array)
        
    Returns:
        - Number of mismatches
    """
    # Convert to numpy arrays if needed
    if not isinstance(grid1, np.ndarray):
        grid1 = np.array(grid1)
    if not isinstance(grid2, np.ndarray):
        grid2 = np.array(grid2)
    
    # Get maximum dimensions
    max_rows = max(grid1.shape[0], grid2.shape[0])
    max_cols = max(grid1.shape[1], grid2.shape[1])
    
    # Pad grids to match maximum size
    padded1 = np.full((max_rows, max_cols), -1, dtype=object)
    padded2 = np.full((max_rows, max_cols), -1, dtype=object)
    
    # Copy original grids into padded arrays
    padded1[:grid1.shape[0], :grid1.shape[1]] = grid1
    padded2[:grid2.shape[0], :grid2.shape[1]] = grid2
    
    # Calculate mismatches
    mismatch_mask = padded1 != padded2
    num_mismatches = np.sum(mismatch_mask)
    
    return num_mismatches #, mismatch_mask

# alternative distance metric, I am not sure it is actually used
def pso_distance(grid1, grid2):
    
    # Canvas area difference
    canvas_diff = abs(height(grid1)*width(grid1) - height(grid1)*width(grid2)) / 900
    
    # Color palette difference
    color_diff = len(palette(grid1) ^ palette(grid2)) / 10  # Using symmetric difference
    
    # Object count difference
    obj_1 = len(objects(grid1, T, F, F))
    obj_2 = len(objects(grid2, T, F, F))
    obj_diff = abs(obj_1/width(grid1)/height(grid1) - obj_2/width(grid2)/height(grid2))

    # Pixel difference
    pixel_diff = grid_distance(grid1, grid2) / 900
    
    return (canvas_diff + color_diff + obj_diff + pixel_diff) / 4

# classes necessary for the fitness function
class Counter:
    def __init__(self):
        self.ex_count = 0
        self.err_count = 0
        self.id_count = 0

    def inc_ex(self):
        self.ex_count += 1
    
    def inc_err(self):
        self.err_count += 1

    def inc_id(self):
        self.id_count += 1

    def get_count(self):
        return self.err_count/self.ex_count, self.id_count/self.ex_count


def fitness(expr, counter, data='train'):

    def try_evaluate_example(fun, example):
        counter.inc_ex()
        try:
            #print(fun(example['input']), example['output'])
            predicted_output = fun(example['input'])
            if grid_distance(example['input'], predicted_output)==0: counter.inc_id()
            return grid_distance(predicted_output, example['output'])
        except Exception as e:
            counter.inc_err()
            # Print the error message
            # print(f"Error evaluating example: {e}")
            # Return a large penalty value for the error
            return 900

    fun = eval('lambda I: ' + expr)    
    fit = sum(try_evaluate_example(fun, example) for example in task[data]) 
    return fit


if __name__ == "__main__" :
    
    # hard-coded values
    random_seed = 42
    max_number_of_keywords = 3
    max_program_depth = 10
    max_function_evaluations = int(1e6)
    population_size_ga = 100
    particle_number_pso = 100
    technique_names = [
        'random_search', 
        'hill_climber', 
        'genetic_algorithm',
        'particle_swarm_optimisation'
                       ]
    repetitions_per_technique = 30
    
    # we need a lot of files for these experiments!
    file_pto_prior = "../data/root_tree.json"
    file_task_keywords = "../data/neural_network_y.csv"
    folder_arc_tasks = "../local/re-arc/arc_original/training/"
    folder_output = "../local/" + os.path.basename(__file__)[:-3]

    # necessary preparations: create folder and initialize logger
    if not os.path.exists(folder_output) :
        os.makedirs(folder_output)    
    
    logger = initialize_logging(folder_output, "log", date=False)
    
    # let's prepare a separate random number generator; PTO uses the global
    # 'random' module, so should not contaminate that
    local_prng = random.Random()
    local_prng.seed(random_seed)
    
    # first, load the table with the keywords used in each task
    logger.info("Loading keyword table file \"%s\"..." % file_task_keywords)
    df = pd.read_csv(file_task_keywords)
    # sort the tasks by sum of number of keywords by row
    numeric_cols = df.select_dtypes(include="number")
    df["row_sum"] = numeric_cols.sum(axis=1)
    df = df.sort_values(by="row_sum", ascending=True)
    # select the tasks
    df_selection = df[df["row_sum"] <= max_number_of_keywords]
    
    target_tasks = df_selection["task_id"].values
    logger.info("Tasks selected with %d keywords or less: %s" % 
                (max_number_of_keywords, str(target_tasks)))
    
    # load prior on program generation
    logger.info("Loading program prior file \"%s\"..." % file_pto_prior)
    node_tree = None
    with open('../data/node_tree.json', 'r') as f:
        node_tree = json.load(f)
        
    # generate a random expression, just to see if it works
    random_program = generate_random_expression(node_tree, max_depth=max_program_depth)
    logger.info("Random program: \"%s\"" % random_program)
    
    # start the experiments!
    for task_id in target_tasks :
        logger.info("Starting experiments on task \"%s\"..." % task_id)
        logger.info("Loading task data...")
        
        with open(os.path.join(folder_arc_tasks, task_id + ".json"), "r") as fp :
            task = utils.format_task(json.load(fp))
            
        logger.info("Task \"%s\" has %d training examples and %d test examples" %
                    (task_id, len(task['train']), len(task['test'])))
        
        # let's setup a dictionary to store the results
        task_dictionary = {
            'technique' : [], 
            'repetition' : [],
            'random_seed' : [],
            'exception_frequency' : [],
            'do_nothing_frequency' : [],
            'train_fitness' : [],
            'test_fitness' : [],
            #'genotype' : [], # for the moment, let's ignore the genotype
            'phenotype' : [],
            'generations' : [],
            'run_time' : []
                           }
        
        # now, we search for a specific file inside the folder; this
        # is the file in which we store all partial results for the techniques
        file_task_results = os.path.join(folder_output, "task_" + task_id + "_results.csv")
        if os.path.exists(file_task_results) :
            df_results = pd.read_csv(file_task_results)
            task_dictionary = df_results.to_dict(orient='list')
        
        for technique_name in technique_names :
            for r in range(0, repetitions_per_technique) :
                
                logger.info("Running \"%s\", repetition %d..." % (technique_name, r))
                
                # generate a random seed for the experiment
                experiment_random_seed = local_prng.randint(0, int(1e6))
                random.seed(experiment_random_seed)
                
                # check if the experiment is already in the dictionary
                existing_repetitions = [r for r in task_dictionary['technique']
                                        if r == technique_name]
                
                # if the experiment already exists, skip it
                if len(existing_repetitions) <= r :
                    
                    # some of the arguments might be technique-specific, so let's
                    # perform some checks
                    solver_args = {}
                    callback_function = None
                    if technique_name == 'random_search' or technique_name == 'hill_climber' :
                        solver_args = {'n_generation' : max_function_evaluations}
                        callback_function = lambda state : ((logger.info("Error: %d; Function calls: %d" % (state[1], state[2])) if state[2]%100==0 else None) or state[1]==0)
                    elif technique_name == 'genetic_algorithm' :
                        n_generation = int(max_function_evaluations / population_size_ga)
                        solver_args = {
                            'n_generation' : n_generation,
                            'population_size' : population_size_ga,
                                       }
                        callback_function = lambda state : ((logger.info("Generation %d: Best fitness %d" % (state[2], state[1][0]))) or state[1][0]==0)
                    elif technique_name == 'particle_swarm_optimisation' :
                        n_iteration = int(max_function_evaluations / particle_number_pso)
                        solver_args = {
                            'n_iteration' : n_iteration,
                            'n_particles' : particle_number_pso
                            }
                        callback_function = lambda state : ((logger.info("Generation %d: Best fitness %d" % (state[2], state[1][0]))) or state[1][0]==0)
                    
                    # run one of the considered algorithms
                    time_start = time.time()
                    (pheno, geno), fx, num_gen = run(
                        generate_random_expression, 
                        fitness, 
                        gen_args=(node_tree,),
                        fit_args=(counter:=Counter(),), 
                        Solver=technique_name, 
                        solver_args={'n_generation': max_function_evaluations}, 
                        callback=callback_function, 
                        better=min
                        )
                    run_time = time.time() - time_start
                    
                    # update dictionary and save to csv
                    task_dictionary['technique'].append(technique_name)
                    task_dictionary['repetition'].append(r)
                    task_dictionary['random_seed'].append(experiment_random_seed)
                    task_dictionary['exception_frequency'].append(counter.get_count()[0])
                    task_dictionary['do_nothing_frequency'].append(counter.get_count()[1])
                    task_dictionary['train_fitness'].append(fx)
                    task_dictionary['test_fitness'].append(fitness(pheno, counter_test:=Counter(), data='test'))
                    #task_dictionary['genotype'].append(str(geno))
                    task_dictionary['phenotype'].append(pheno)
                    task_dictionary['generations'].append(num_gen)
                    task_dictionary['run_time'].append(run_time)
                    
                    df_results = pd.DataFrame.from_dict(task_dictionary)
                    df_results.to_csv(file_task_results, index=False)
                else :
                    logger.info("Already found results for task \"%s\", technique \"%s\", repetition %d; skipping to the next experiment..."
                                % (task_id, technique_name, r))
                    
            # end of the loop for each repetition
            
        # end of the loop for each technique
        
        # end of the task
        
        # TODO: remove this, it's just debugging
        sys.exit(0)
        
    # finish the experiments
    close_logging(logger)

    
    