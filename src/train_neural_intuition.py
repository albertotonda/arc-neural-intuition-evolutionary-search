# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 13:22:44 2024

This script will attempt to train the neural networks.

@author: Alberto
"""
import json
import os
import numpy as np
import pandas as pd
import torch

# local imports
import common
import neural_networks

if __name__ == "__main__" :
    
    # some hard-coded values
    class_names_file = "../data/neural_network_class_labels.csv"
    y_labels_file = "../data/neural_network_y.csv"
    arc_tasks_folder = "../data/re_arc/tasks"
    percentage_of_samples_for_training = 0.8
    percentage_of_samples_for_validation = 0.1
    # hyperparameters of the optimizer
    learning_rate = 1e-7
    n_epochs = 1000
    
    # this is a list of the training tasks
    arc_task_ids = [
        "3c9b0459",
        "6150a2bd",
        "67a3c6ac",
        "68b16354",
        "74dd1130",
        "a416b8f3",
        "ed36ccf7"
        ]
    
    # the real script starts here
    logger = common.initialize_logging("../local", "train_neural_intuition")
    logger.info("Considering a subset of %d tasks: %s" % 
                (len(arc_task_ids), str(arc_task_ids)))
    logger.info("Preparing training data...")
    
    # load everything into dataframes
    logger.info("Loading \"%s\"..." % class_names_file)
    df_class_names = pd.read_csv(class_names_file)
    logger.info("Loading \"%s\"..." % y_labels_file)
    df_y_labels = pd.read_csv(y_labels_file)
    
    # select only tasks we are interested in
    df_y = df_y_labels[df_y_labels["task_id"].isin(arc_task_ids)]
    # and then, select only columns (features) which are non-empty
    non_zero_columns = [c for c in df_y.columns if (df_y[c] != 0).any()]
    keywords_used = [c for c in non_zero_columns if c != "task_id"]
    df_y = df_y[non_zero_columns]
    
    # print out some statistics
    n_keywords = len(keywords_used)
    logger.info("There are a total of %d keywords used in this subset of tasks: %s" %
                (n_keywords, str(keywords_used)))
    
    # now, let's take a look at the folder with all the extra training tasks
    # created by Hodel; they are stored as .json files
    task_files = [os.path.join(arc_tasks_folder, t) for t in os.listdir(arc_tasks_folder)
                  if t.endswith(".json") and t[:-5] in arc_task_ids]
    logger.info("Found %d task files: %s" % (len(task_files), str(task_files)))
    
    # load all the necessary data
    n_samples = 0
    task_input_output_pairs = dict()
    for i, task_id in enumerate(arc_task_ids) :
        task_input_output_pairs[task_id] = common.load_arc_task_from_json(task_files[i])
        logger.info("For task \"%s\", found %d training pairs!" %
                    (task_id, len(task_input_output_pairs[task_id]["input"])))
        n_samples += len(task_input_output_pairs[task_id]["input"])
        
    # and now, here starts the boring preprocessing part; we have to create a
    # huge numpy array (n_samples, 1, 30, 30) for the inputs and for the outputs,
    # and then create a corresponding array of labels; we use a -1 for the initial
    # value, because '0' is an existing value among the grids, which has a meaning
    input_grids = -1. * np.ones((n_samples, 1, 30, 30))
    output_grids = -1 * np.ones((n_samples, 1, 30, 30))
    y_labels = np.zeros((n_samples, n_keywords))
    
    sample_index = 0
    for task_id, pairs in task_input_output_pairs.items() :
        
        # the class labels for samples of this task are always going to be the same;
        # we are transforming the query to the DataFrame into a one-dimensional array
        class_labels = df_y[df_y["task_id"] == task_id].values[0,1:]
        
        for i in range(0, len(pairs["input"])) :
            # get the grids out of the dictionary
            input_grid = pairs["input"][i]
            output_grid = pairs["output"][i]
            #logger.info(str(input_grid.shape))
            
            # place the grids into the appropriate place inside the future tensors;
            # as the grids are MAXIMUM 30x30, place them with the proper offset
            start_x = (30 - input_grid.shape[0]) // 2
            start_y = (30 - input_grid.shape[1]) // 2
            input_grids[sample_index,0,start_x:start_x + input_grid.shape[0], start_y:start_y + input_grid.shape[1]] = input_grid
            
            start_x = (30 - output_grid.shape[0]) // 2
            start_y = (30 - output_grid.shape[1]) // 2
            output_grids[sample_index,0,start_x:start_x + output_grid.shape[0],start_y:start_y + output_grid.shape[1]] = output_grid
            
            y_labels[sample_index,:] = class_labels
            
            sample_index += 1
            
    logger.info("y_labels: %s" % str(y_labels))
    logger.info("input_grids[0]: %s" % str(input_grids[0]))
    
    # we can now normalize everything in 0,1; we know min and max
    input_grids = (input_grids+1) / (9+1)
    output_grids = (output_grids+1) / (9+1)
    logger.info("normalized input_grids[0]: %s" % str(input_grids[0]))
    
    
    # cool, now we have to continue the pre-processing part, creating a Dataset
    # and a DataLoader specific to our problem; however, we could also first
    # separate the dataset between training and test in a stratified fashion
    n_training_samples = int(percentage_of_samples_for_training * n_samples)
    n_validation_samples = int(percentage_of_samples_for_validation * n_samples)
    n_test_samples = n_samples - n_training_samples - n_validation_samples
    logger.info("Of the original %d samples: train=%d, validation=%d, test=%d" %
                (n_samples, n_training_samples, n_validation_samples, n_test_samples))
    logger.info("Splitting data into training/validation/test...")
    
    # there might be more intricate ways of splitting the data, but for the moment
    # we just take them in-order
    train_input_grids = np.zeros((n_training_samples, 1, 30, 30))
    train_output_grids = np.zeros((n_training_samples, 1, 30, 30))
    y_train = np.zeros((n_training_samples, n_keywords))
    
    val_input_grids = np.zeros((n_validation_samples, 1, 30, 30))
    val_output_grids = np.zeros((n_validation_samples, 1, 30, 30))
    y_val = np.zeros((n_validation_samples, n_keywords))    
    
    test_input_grids = np.zeros((n_test_samples, 1, 30, 30))
    test_output_grids = np.zeros((n_test_samples, 1, 30, 30))
    y_test = np.zeros((n_test_samples, n_keywords))  
    
    # fill in the training/validation/test sets, iterating over tasks,
    # so that a certain amount of samples from each task end up in each set
    start_task_index = 0
    train_index = 0
    val_index = 0
    test_index = 0
    for task_id, pairs in task_input_output_pairs.items() :
        n_task_samples = len(pairs["input"])
        n_task_train = int(n_task_samples * percentage_of_samples_for_training)
        n_task_val = int(n_task_samples * percentage_of_samples_for_validation)
        n_task_test = n_task_samples - n_task_train - n_task_val
        
        task_index = start_task_index
        
        train_input_grids[train_index:train_index+n_task_train,:,:,:] = \
            input_grids[task_index:task_index+n_task_train,:,:,:]
        train_output_grids[train_index:train_index+n_task_train,:,:,:] = \
            output_grids[task_index:task_index+n_task_train,:,:,:]
        y_train[train_index:train_index+n_task_train,:] = \
            y_labels[task_index:task_index+n_task_train]
        
        task_index += n_task_train
        
        val_input_grids[val_index:val_index+n_task_val,:,:,:] = \
            input_grids[task_index:task_index+n_task_val,:,:,:]
        val_output_grids[val_index:val_index+n_task_val,:,:,:] = \
            output_grids[task_index:task_index+n_task_val,:,:,:]
        y_val[val_index:val_index+n_task_val,:] = \
            y_labels[task_index:task_index+n_task_val]
        
        task_index += n_task_val
        
        test_input_grids[test_index:test_index+n_task_test] = \
            input_grids[task_index:task_index+n_task_test]
        test_output_grids[test_index:test_index+n_task_test] = \
            output_grids[task_index:task_index+n_task_test]
        y_test[test_index:test_index+n_task_test] = \
            y_labels[task_index:task_index+n_task_test]
        
        # jump to the samples of the next task; also update the indexes for
        # the other matrices
        task_index += n_task_samples
        train_index += n_task_train
        val_index += n_task_val
        test_index += n_task_test
        
    # now we can finally create the Dataset and the DataLoader for this task
    logger.info("Generating pytorch Datasets and DataLoaders...")
    train_data = neural_networks.GridPairsDataset(train_input_grids, train_output_grids, y_train)
    val_data = neural_networks.GridPairsDataset(val_input_grids, val_output_grids, y_val)
    test_data = neural_networks.GridPairsDataset(test_input_grids, test_output_grids, y_test)
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=len(val_data), shuffle=False)
    test_data = torch.utils.data.DataLoader(test_data, batch_size=len(test_data), shuffle=False)
    
    # instantiate the network, and start training
    model = neural_networks.SiameseMultiLabelNetwork(n_classes=n_keywords, input_channels=1)
    loss = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # training loop
    for epoch in range(n_epochs) :
        # iterate over the batches
        train_loss = 0.0
        for input_grid, output_grid, y_sample in train_loader :
            #G1, G2, labels = G1.to(device), G2.to(device), labels.to(device)
            #logger.info(str(input_grid.shape))
            
            optimizer.zero_grad()
            outputs = model(input_grid, output_grid)
            batch_loss = loss(outputs, y_sample)
            batch_loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += batch_loss.item()
            
        # at the end of the batches, compute train loss
        train_loss = train_loss / len(train_loader)
        
        # also evaluate validation loss
        with torch.no_grad() :
            val_loss = 0.0
            for input_grid, output_grid, y_sample in val_loader :
                outputs = model(input_grid, output_grid)
                val_loss = loss(outputs, y_sample)
        
        logger.info("Epoch %d/%d, loss: training=%.6f, val=%.6f" %
                    (epoch, n_epochs, train_loss, val_loss))
    
    common.close_logging(logger)