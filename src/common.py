# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 08:06:18 2024

Common library with utility functions for logging and pre-processing ARC tasks

@author: Alberto Tonda
"""
import datetime
import json
import logging
import os
import numpy as np

def initialize_logging(path: str, log_name: str = "", date: bool = True) -> logging.Logger :
    """
    Function that initializes the logger, opening one (DEBUG level) for a file 
    and one (INFO level) for the screen printouts.
    """

    if date:
        log_name = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "-" + log_name
    log_name = os.path.join(path, log_name + ".log")

    # create log folder if it does not exists
    if not os.path.isdir(path):
        os.mkdir(path)

    # remove old logger if it exists
    if os.path.exists(log_name):
        os.remove(log_name)

    # create an additional logger
    logger = logging.getLogger(log_name)

    # format log file
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("[%(levelname)s %(asctime)s] %(message)s",
                                  "%Y-%m-%d %H:%M:%S")

    # the 'RotatingFileHandler' object implements a log file that is automatically limited in size
    fh = logging.handlers.RotatingFileHandler(log_name,
                             mode='a',
                             maxBytes=100*1024*1024,
                             backupCount=2,
                             encoding=None,
                             delay=0)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # add an INFO-level handler for the console
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    logger.info("Starting " + log_name + "!")

    return logger

def close_logging(logger: logging.Logger) :
    """
    Simple function that properly closes the logger, avoiding issues when the program ends.
    """

    for handler in logger.handlers:
        handler.close()
        logger.removeHandler(handler)

    return

def load_arc_task_from_json(json_file : str) :
    """
    Load an ARC task from a JSON file, expecting the usual structure: a list
    of dictionaries, each with an "input" and an "output" key. Let's transform
    it into a list of inputs and a list of outputs.
    """
    data = json.load(open(json_file, "r"))
    output_dictionary = {"input" : [], "output" : []}
    
    for pair in data :
        output_dictionary["input"].append(np.array(pair["input"]))
        output_dictionary["output"].append(np.array(pair["input"]))
        
    return output_dictionary
        
if __name__ == "__main__" :
    
    logger = initialize_logging("../local", "my_log", date=True)
    logger.info("This is an INFO-level log message, which will appear both on screen and in the log file")
    logger.debug("And this is a DEBUG level log message, which will only appear in the log file")
    
    arc_task_file = "../data/re_arc/tasks/3c9b0459.json"
    logger.info("Now, let me try reading an ARC task from a JSON file")
    arc_task = load_arc_task_from_json(arc_task_file)
    print("Found %d inputs and %d outputs!" % 
          (len(arc_task["input"]), len(arc_task["output"])))
    
    close_logging(logger)
    
    