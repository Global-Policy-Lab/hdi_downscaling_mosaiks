# This file holds solve functions related to the autotune process
import gc
import glob
import itertools
import os
import pickle
import shutil
import time
import copy

import numpy as np
import pandas as pd

from functools import reduce
import sklearn.metrics as metrics
from mosaiks import config as c
from mosaiks.utils.logging import log_text, robust_chmod
from mosaiks.solve import data_parser as parse
from mosaiks.solve import interpret_results as ir

from mosaiks.solve import solve_functions as solve
from mosaiks.solve import master_solve as ms

from mosaiks.utils import io as mio
from mosaiks.utils import config_read

import mosaiks.plotting.general_plotter as plots

DEBUG = False




def get_truth_preds(kfold_results):
    # code is flexible for looping over lists of results (for regional models)--
    # if a single set of results is passed, it is coerced into a list for
    # compatibility.
    if not isinstance(kfold_results, list):
        kfold_results = [kfold_results]
    
    # pull out the results
    truth = np.vstack(
        [np.vstack([solve.y_to_matrix(i) for i in r["y_true_test"].squeeze()])
         for r in kfold_results]
    )
    
    best_preds = np.vstack(
        [ir.interpret_kfold_results(r, crits="r2_score")[2] for r in kfold_results]
    )
    preds = np.vstack([solve.y_to_matrix(i) for i in best_preds.squeeze()])
    
    return truth, preds

def get_r2(kfold_results):
    """
    Returns r2 from truth and preds. 
    
    If truth and preds are strings (meaning we are using the classifier) returns accuracy
    """
    truth, preds = get_truth_preds(kfold_results)
    # To deal with classifier, we check for strings in truth, preds

    #TODO better way to write this conditional should be possible
    if isinstance(truth[0][0], str):
        accuracy = (truth == preds).sum()/len(truth)
        return accuracy

    else:
        r2 = metrics.r2_score(truth, preds)
        return r2

def check_region_counts(regions, threshold=20):
    """
    Checks for a minimum number of observations in each region group. If minimum is not met, we cannot run a region model
    
    regions: np array containing string region values
    threshold: the minimum number of observations needed to run a regional model
    
    Returns: bool
    """
    
    regions = pd.Series(regions, name="regions").reset_index()
    regions_grouped = regions.groupby("regions").count()

    print(regions_grouped) 
    
    return not any(regions_grouped.values < threshold)



def replace_X_with_weighted(c, c_app, locations, regions, pop_weight=True):
    """
    locations: polygon_ids (note that this function only works on polygon aggregated outcomes)
    pop_weight: pop_weight param passed into c_app. See README.MD for full set of options (can be log pop weight, for example)
    regions: pd.dataframe or pd.series object with region string identifiers. The index must be polygon_ids (locations)
    """
    c_app = copy.deepcopy(c_app)
    c_app["pop_weight"] = pop_weight
    polygon_id_colname = c_app["polygon_id_colname"]

    c_app["region_type"] = None #No need to get region identifier again, we will subset
    X_pop, _, y_pop, _  = mio.get_X_locations_y(c, c_app, polygon_id_colname)
    
    try:    
        X_pop_train = X_pop.loc[locations.flatten()]
        y_pop_train = y_pop.loc[locations.flatten()] #NOTE that this should be identical to Y_train
        regions_pop_train = regions.loc[locations.flatten()] #NOTE that this should be identical to regions_train

        log_text("pop weighted X locations could be obtained for all rows of X_train")
        log_text("Implies that locations and y are identical in length")

        return np.array(X_pop_train), np.array(locations), np.array(y_pop_train), np.array(regions_pop_train)


    except:
        locs_available = pd.Series(locations.flatten()).isin(X_pop.index)
        new_locations = locations[locs_available]
        missing_Xs = len(locations) - len(new_locations)

        log_text("Train idxs for area weight X are not in pop weight X", print_text="warn")
        log_text(f"Missing {missing_Xs} rows")
        log_text(f"These observations will be droppd in the pop weight model")

        X_pop_train = X_pop.loc[new_locations.flatten()]
        y_pop_train = y_pop.loc[new_locations.flatten()]
        regions_pop_train = regions.loc[new_locations.flatten()] #We drop the missing observations from regions array

        return np.array(X_pop_train), np.array(new_locations), np.array(y_pop_train), np.array(regions_pop_train)


def move_pred_to_main(c, app, results, summary_tbl, max_key):
    '''
    Helper function to copy best predictions to main folder
    Also deletes old predictions from main for the same app name
    '''

    # Delete old files
    for data in glob.glob(c.main_pred_dir + '/*' +  app +  '*'):
        log_text('remove ' + data)
        os.remove(data)

    # Move and copy
    new_pred_path = results[max_key]['file_path']['pred'].replace(c.pred_savedir, c.main_pred_dir)  
    shutil.copy(results[max_key]['file_path']['pred'], new_pred_path)
    
    results[max_key]['file_path']['pred'] = new_pred_path
    summary_tbl['file_path_pred'][summary_tbl['run_name'].index(max_key)]  =  new_pred_path

    return results, summary_tbl

def move_plot_to_main(c, app, results, summary_tbl, max_key):
    '''
    Helper function to copy best plots to main folder
    Also deletes old predictions from main for the same app name
    '''

    # Delete old files
    for data in glob.glob(c.main_plot_dir + '/*' +  app +  '*'):
        log_text('remove ' + data)
        os.remove(data)

    for data in glob.glob(results[max_key]['file_path']['plot']):
        new_plot_path = data.replace(c.plot_savedir, c.main_plot_dir)  
        shutil.copy(data, new_plot_path)
        robust_chmod(new_plot_path)
    
    summary_tbl['file_path_plot'][summary_tbl['run_name'].index(max_key)]  =  results[max_key]['file_path']['plot'].replace(c.plot_savedir, c.main_plot_dir)
    results[max_key]['file_path']['plot'] = results[max_key]['file_path']['plot'].replace(c.plot_savedir, c.main_plot_dir)

    return results, summary_tbl

def neg_condition(x):
    return x<0

def tune_model(
            app,
            from_meta=True,
            tunelambda="coarse",
            save_all=True,
            return_all=False,
            plot_all=False,
            overwrite = False,
            DEBUG=False,
            VERBOSE = True,
            ):

    '''
    This function runs all feasible combination of tuning models:
    Stage 1
        - Automatically extracts from csv, adds an intercept, and If polygon only do area weighting
        - If more than 10% of the data are 0s, runs hurdle model.
        - If the data spans more than one region, runs the region specific model.
        - Tries both log and levels
        - The best model is chosen based off of r2. If the best model uses log, checks that it is at least 
            10% better than the levels model. If not, the best model will be changed to the levels.
    Stage 2:
        - Uses the “best” model from stage 1 (i.e. whether or not to use log or levels and hurdle or not hurdle)
        - Tries both with and without an intercept
        - Tries both bounded and unbounded predictions.
        - If the data spans more than one region, tries both a region specific and pooled model.
        - If the data are polygons, runs both population and area weighted feature aggregations.

    Parameters:
    ----------
        app (string): The name of your config (must have a config of the same name in
            either `mosaiks/configs/{app_folder}/` or in `mosaiks/configs`.)
        from_meta (bool): Load the data needed for a config from the metadata file
        tune_lambda ("coarse", "fine", or False): String to tune lambdas, otherwise False.
        save_all (bool): If true, save all predictions, parameters, and metrics to disk.
        overwrite (bool): If True, overwrite existing files. Else, any existing files are read.
        return_all (bool): If true, returns all results. If False, returns summary tables.
        plot_all (bool): If true, plots prediction results for each tuning iteration and saves
            param to random sample if global use generic lambdas or set lambdas
        DEBUG (bool): If true, subsets data to only 1000 entries
        VERBOSE (bool): If true, prints statements to output


    Returns:
    ----------
        results1 (dict): Only returns if return_all is True. Dictionary of all kfold results for every 
            model run in model tuning
        summary_tbl1 (dict): Summary dictionary for stage 1 with the following keys: transformation, 
            hurdle, r2, file_path_pred (file path for predictioon storage), file_path_plot (file path for plot
            storage), run_name, best_model (boolean to designate best model based on criteria).
        results2 (dict): Only returns if return_all is True. Dictionary of all kfold results for every 
            model run in parameter tuning
        summary_tbl2 (dict): Summary dictionary for stage 2 tuning 
    '''
    start_time = time.time()
    ###########################################
    #### Initialize model                   ###
    ###########################################
     


    # Check for strange function inputs
    if overwrite and not save_all:
        log_text( ("overwrite=True and save_all=False \n"
         "Autotune will be run all for model combinations but outputs will not be saved \n"
         "This is not typically desirable behavior..."), 
         print_text="warn"
        )
    if not overwrite and save_all:
        log_text( ("overwrite=False and save_all=True \n"
         "Previous models from autotune will be read in if they are available. New summary tables \n"
         "will be saved. This overwrites previous summary tables ...") 
        )
    # Check for depracated input
    if tunelambda not in ["coarse", "fine", False, None]:
        log_text("tunelambda argument not understood and is likely deprecated", print_text="warn")
        log_text("Defaulting to coarse lambda tuning")
        tunelambda = "coarse"
    
    ## IMPLEMENT INITIAL PARAMS/DEFAULT BEHAVIOR
    (c, c_app, outcome_name,
     labels_file, grid, 
     polygon_id_colname) = config_read.extract_config(app=app,
                                                    from_meta=from_meta, auto_tune=True)

    log_text(f"Extracted c_app:  {c_app}")

    c.verbose = VERBOSE #use model param to override setting in c
    
    c_app["tunelambda"] = tunelambda #Pass optional arg to function into c_app

    region_type = copy.deepcopy(c_app.get("region_type","continent")) #Get region type or default to continent
    
    # We only default to continent regions for autotuning purposes

    model = copy.deepcopy(c_app.get("solve_function", "Ridge"))
    classifier = (model == "OVR_classifier")
    
    # FEATURE AGGREGATION WEIGHTS
    if polygon_id_colname:
        log_text("Polygon, default behavior to run area agg")
        
        # Set population weighting to false
        c_app['pop_weight_features'] = False

        pop_weight_feats_created = False
    
    # setting region_type allows us to get the appropriate regions
    # information, which will allow us to determine whether we actually
    # want to run a region model in the autotune. 
    c_app["region_type"] = region_type 

    X, locations, y, regions = mio.get_X_locations_y(c, c_app, 
                                                     polygon_id_colname)
    
    # create empty results dictionary
    # a dictionary with unique strings that correspond to each "branch". 
    # E.g., `log_hurdle_no_clip_kfold_results`
    results1 = {} 
    
    # Merge, split data

    # Grab train/test split
    (
    X_train,
    _,
    Y_train,
    _,
    locations_train,
    _,
    regions_train,
    _
    ) = parse.merge_dropna_split_train_test(c, app, c_app, labels_file, X,
                                            locations, y, regions)
     # Truncate for testing
    if DEBUG:
        X_train = X_train[:1000] 
        Y_train = Y_train[:1000]
        locations_train = locations_train[:1000]
        regions_train = regions_train[:1000]

    # We no longer use hurdle in autotune, we keep given solve function (with "Ridge" default)
    c_app["solve_function"] = model
    
    # LOG or LEVELS
    if classifier:
        transformation_options = [None]
    else:
        transformation_options = [None, "log"]
    
    # Run regions?
    num_regions = len(np.unique(regions_train))
    log_text("unique regions: \n {}".format(np.unique(regions_train)))
    
    # we want a lower min obs threshold for countries than for continents

    min_obs_in_all_regions = check_region_counts(
        regions_train, threshold=20)

    log_text("min_obs_in_all_regions: {}".format(min_obs_in_all_regions) )
    # If your labeled data extends across >1 region, use a region-specific model
    # We also have a minumum number of observations on each region needed for a continent model
    # For a country model, we do not have a minuimum. This means that observations may be dropped later
    #While this is undesirable, it allows the code to be less susceptible to breaking in small countries

    if (num_regions > 1) and (min_obs_in_all_regions or (region_type=="country")):
        log_text("\n Using region specific model")
        run_regions = True

    else:
        log_text("\n Using pooled region model")
        run_regions = False

    if run_regions:
        c_app["region_type"] = region_type
    else:
        c_app["region_type"] = None


    ### We clip to observed min, and max of train data
    ### We do do this for all models in autotune
    c_app['bounds_pred'] = "auto"

    ### Set remaining first stage defaults
    ## In first stage, we area weight polygons
    c_app['pop_weight_features'] = False

    #We start with an intercept in the first stage
    c_app['intercept'] = True
    
    del X, y, locations
    gc.collect()
    
    ###########################################
    #### Stage 1.                           ###
    ###########################################
    log_text("\nSTAGE 1: Finding best model between log/level...")    

    for t_op in transformation_options:
        
        if t_op == None:
            t_op_str = 'lvls'
        else:
            t_op_str = t_op
        
        # Transform data if needed
        c_app['transformation'] = t_op

        h_op_str = '_nohurdle'
        
        log_text("\nTry transformation {} with {}".format(t_op, model)) 
            
        c = mio.get_filename(c, app, c_app, labels_file)


        results1[t_op_str + h_op_str] = {}
        file_path = c.pred_savedir + '/' + c.fname + '.pickle'

        results1[t_op_str + h_op_str]['file_path'] = {}
        if save_all:
            results1[t_op_str + h_op_str]['file_path']['pred'] =  file_path
        if plot_all:
            results1[t_op_str + h_op_str]['file_path']['plot'] = c.plot_savedir + '/*' + c.fname + '*'
        results1[t_op_str + h_op_str]['params'] = {'transformation': t_op,
                                                'hurdle': False}
        create_or_overwrite = (overwrite or not os.path.exists(file_path))

        if create_or_overwrite:
            results1[t_op_str + h_op_str]['results'] = ms.master_kfold_solve(
                c, c_app, X_train, Y_train, locations_train, regions_train
            )

        else:
            log_text("Output file exists for solve")
            log_text("Reading existing file... \n")
            results1[t_op_str + h_op_str]['results'] = pd.read_pickle(file_path)

                                                                            
        # Plot results
        if plot_all:
            plots.plot_diagnostics(results1[t_op_str +  h_op_str]['results'], 
                                outcome_name, 
                                app,
                                polygon_id_colname, 
                                grid, 
                                c, 
                                c_app)
            
        if save_all and create_or_overwrite:
            with open(file_path, 'wb') as f:
                pickle.dump(results1[t_op_str +  h_op_str]['results'], f)

            robust_chmod(file_path)

    # Grab and add r2 to reslts dictionary
    # In same loop, create and summary table
    summary_tbl1 = {'transformation': [], 'hurdle': [], 
                    'r2': [], 'file_path_pred': [],
                    'file_path_plot': []}
    
    for key in results1.keys():
        results1[key]['r2'] = get_r2(results1[key]['results'])
        
        summary_tbl1['transformation'].append(results1[key]['params']['transformation'])
        summary_tbl1['hurdle'].append(results1[key]['params']['hurdle'])
        summary_tbl1['r2'].append(results1[key]['r2'])
        
        if save_all:
            summary_tbl1['file_path_pred'].append(results1[key]['file_path']['pred'])
        if plot_all:
            summary_tbl1['file_path_plot'].append(results1[key]['file_path']['plot'])
    max_key = max(results1, key=(lambda key: results1[key]['r2']))
    
    # If it's a log model check if its at least 10% better than levels
    if results1[max_key]['params']['transformation'] == 'log' and len(transformation_options)>1:
        max_hurdle = results1[max_key]['params']['hurdle']
        
        # Look for other models that have the same model parameters
        compare_key = [k for k, v in results1.items() if v['params']['transformation']==None and v['params']['hurdle'] == max_hurdle].pop()
        
        
        max_r2 = results1[max_key]['r2']
        compare_r2 = results1[compare_key]['r2']
            
        pct_diff = (max_r2 - compare_r2)/compare_r2 *100
        if pct_diff < 10.:
            max_key = compare_key
            
    # WHICH WAS OUR BEST MODEL?   
    # Find our best model with the best r2
    summary_tbl1['run_name'] = list(results1.keys())
    summary_tbl1['best_model'] = np.zeros(len(results1.keys()))
    summary_tbl1['best_model'][summary_tbl1['run_name'].index(max_key)]  = 1
    
    # Move best models to main folder
    if save_all:
        results1, summary_tbl1 = move_pred_to_main(c, app, results1, summary_tbl1, max_key)
        if plot_all:
            results1, summary_tbl1 = move_plot_to_main(c, app, results1, summary_tbl1, max_key)

    
    # Save in 
    stage_1_path = c.pred_savedir + '/' +  app  + '_tuning_stage1_summary.pickle'

    if save_all:
        with open(stage_1_path, 'wb') as f:
            pickle.dump(summary_tbl1, f)
    
    robust_chmod(stage_1_path)
    
    
    ###########################################
    #### Stage 2                            ###
    ###########################################
    log_text("\nSTAGE 2 model tuning...") 
    
    # Now with our best model, tune our individual tuning parameters (if specified)
    # aggregation, clipping, regional model, intercept
    
    # For regions: try both region specific model and pooled model if data
    # spans more than one region

    #TODO the best model from stage 1, with default stage 1 params, gets run again below. This could be changed.
    
    if region_type == "country":
        region_options = [region_type]

    elif run_regions:
        region_options = [region_type, None]

    else:
        # If just one region continue to just use pooled model
        region_options = [None]

    # Aggregation
    if polygon_id_colname:
        log_text('Since polygon, try population weighting and area weighitng')
        pop_weight_feat_options = [False,  True] 
    else:
        pop_weight_feat_options= [False] #This param does not actually change anything at tile level
    
    

    # Create keys for params sub dictionary 
    dict_sub_keys = {
                    'pop_weight_features': { True: 'popweight', False: 'areaweight'}, 
                    'region_type': {"country": 'country', False: 'pool', None: "pool", "continent":"continent"},
                    'intercept': {True: 'intercept', False: 'nointercept'}}

    log_text('Testing parameters...') 

    # Set model options to be that of the best model and re-transform training data
    c_app['transformation'] = results1[max_key]['params']['transformation']
    
    hurdle = results1[max_key]['params']['hurdle']
    
    c_app["solve_function"] = model
        
    log_text("From step 1 analysis: model = {} and log = {}".format(model,c_app['transformation']))
    log_text("\n")
    
    # Get list of all unique combinations    
    param_change = list(itertools.product(                       
                                          pop_weight_feat_options, # aggregation options 
                                          region_options, # regional model options
                                          [True, False])) # intercept options                       
    results2 = {}
    
    for j, param in enumerate(param_change):
        # Update c_app based on parameters needed
        c_app['pop_weight_features'] = param[0]
        c_app["region_type"] = param[1]
        c_app['intercept'] = param[2]

        c = mio.get_filename(c, app, c_app, labels_file)
        
        log_text('\nTuning params try {} out of {} \n'.format(j+1, len(param_change))) 
        log_text('Pop weight aggregation: {} \nRun regions: {} \nIntercept: {}'.format(param[0],param[1],param[2])) 

        
        # Create dictionary key for parameter tuning
        dict_key = '_'.join([
                            dict_sub_keys['pop_weight_features'][param[0]],
                            dict_sub_keys['region_type'][param[1]],
                            dict_sub_keys['intercept'][param[2]]]
                           )
        
        # add params to dictionary
        results2[dict_key] = {}
        results2[dict_key]['params'] = {'transformation': c_app['transformation'],
                                        'pop_weight_features': c_app['pop_weight_features'],
                                        'region_type': c_app["region_type"],
                                        'intercept': c_app['intercept']
                                       }

        file_path = c.pred_savedir + '/' + c.fname + '.pickle'
        
        create_or_overwrite = (overwrite or not os.path.exists(file_path))

        results2[dict_key]['file_path'] = {}
        if save_all:
            results2[dict_key]['file_path']['pred'] =  file_path
        if plot_all:
            results2[dict_key]['file_path']['plot'] = c.plot_savedir + '/*' + c.fname + '*'
        
        if create_or_overwrite:
            if not param[0]:
                results2[dict_key]['results'] = ms.master_kfold_solve(c, c_app,
                                                                    X_train,
                                                                    Y_train,
                                                                    locations_train,
                                                                    regions_train)
                
            # If we have more than one aggregation option we're trying, we'll need to get a new X:
            elif param[0] == True:
                if not pop_weight_feats_created: #Make sure we only ever get the weighted Xs one time
                    log_text("Creating pop weighted features")
                    X_train_pop_weight, locations_train_pop_weight, y_train_pop_weight, regions_train_pop_weight = replace_X_with_weighted(
                        c=c,c_app=c_app,locations=locations_train, 
                        regions = regions, pop_weight=True)

                    pop_weight_feats_created = True


                results2[dict_key]['results'] = ms.master_kfold_solve(
                    c, c_app, X_train_pop_weight, y_train_pop_weight,
                    locations_train_pop_weight, regions_train_pop_weight
                )
        else:
            log_text("Output file exists for solve")
            log_text("Reading existing file... \n")
            results2[dict_key]['results'] = pd.read_pickle(file_path)

            
        # Plot results
        if plot_all:
            plots.plot_diagnostics(results2[dict_key]['results'], 
                                outcome_name, 
                                app,
                                polygon_id_colname, 
                                grid, 
                                c, 
                                c_app)

        if save_all and create_or_overwrite:
            with open(file_path, 'wb') as f:
                pickle.dump(results2[dict_key]['results'], f)
            robust_chmod(file_path)   

            
            
            
    # find best model
    # Create summary pickle
    summary_tbl2 = {'transformation': [], 
                    'hurdle': [], 
                    'bound_replace_None_w_observed': [],
                    'pop_weight_features':[],
                    'region_type': [],
                    'intercept': [],
                    'r2': [], 
                    'file_path_pred': [],
                    'file_path_plot': []}


    for key in results2.keys():
        results2[key]['r2'] = get_r2(results2[key]['results'])

        summary_tbl2['transformation'].append(results2[key]['params']['transformation'])
        summary_tbl2['pop_weight_features'].append(results2[key]['params']['pop_weight_features'])
        summary_tbl2['region_type'].append(results2[key]['params']['region_type'])
        summary_tbl2['intercept'].append(results2[key]['params']['intercept'])
        summary_tbl2['r2'].append(results2[key]['r2'])

        if save_all:
            summary_tbl2['file_path_pred'].append(results2[key]['file_path']['pred'])
        if plot_all:
            summary_tbl2['file_path_plot'].append(results2[key]['file_path']['plot'])
    max_key2 = max(results2, key=(lambda key: results2[key]['r2']))


    # Best model? --> find best r2
    summary_tbl2['run_name'] = list(results2.keys())
    summary_tbl2['best_model'] = np.zeros(len(results2.keys()))
    summary_tbl2['best_model'][summary_tbl2['run_name'].index(max_key2)]  = 1    

    
    # Move best models to main folder
    if save_all:
        results2, summary_tbl2 = move_pred_to_main(c, app, results2, summary_tbl2, max_key2)
        if plot_all:
            results2, summary_tbl2 = move_plot_to_main(c, app, results2, summary_tbl2, max_key2)

    stage_2_path = c.pred_savedir + '/' + app  + '_tuning_stage2_summary.pickle'

    if save_all:
        with open(stage_2_path, 'wb') as f:
            pickle.dump(summary_tbl2, f)

    robust_chmod(stage_2_path)

    end_time = time.time()

    log_text("Wall time elapsed (minutes) = {:.0f}".format( (end_time-start_time)/60))
    # Return all results
    if return_all:
        return results1, summary_tbl1, results2, summary_tbl2
    # Return only summary tables
    else:        
        return summary_tbl1, summary_tbl2
    
