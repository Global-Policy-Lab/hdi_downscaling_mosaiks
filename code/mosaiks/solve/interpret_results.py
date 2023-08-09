import warnings
import os
import numpy as np
import glob
import re
import sklearn.metrics as metrics
from mosaiks.solve import data_parser as parse
from mosaiks.solve import solve_functions
from mosaiks.utils.logging import log_text
from mosaiks import config as c
from mosaiks.utils.config_read import parse_metadata

import pandas as pd


def find_best_hp_idx(
    kfold_metrics_outcome, hp_warnings_outcome, crit, minimize=False, val=None, suppress_warnings= False,
):
    """Find the indices of the best hyperparameter combination,
    as scored by 'crit'.
    
    Args:
        kfold_metrics_outcome (n_folds X n_outcomes x ... ndarray of dict (hps are last)): 
            Model performance array produced by kfold_solve for a
            single outcome -- so n_outcomes must be 1
        hp_warnings_outcome (nfolds x n_hps) bool array of whether an hp warning occured
        crit (str): Key of the dicts in kfold_metrics that you want
            to use to score the model performance
        minimize (bool): If true, find minimum of crit. If false (default)
            find maximum
        val (str or int) : If not None, will tell you what outcome is being evaluated
            if a warning is raised (e.g. hyperparameters hitting search grid bounds)
        suppress_warnings (bool): If True, warning that hyperparam is at edge of range will be suppressed
            
    Returns:
        tuple of int: Indices of optimal hyperparameters. 
            Length of tuple will be equal to number of
            hyperparameters, i.e. len(kfold_metrics_test.shape) - 2
    """
    if suppress_warnings:
        sup = False
    else:
        sup = "warn"

    if minimize:
        finder = np.min
    else:
        finder = np.max

    # allowable idxs is wherever a None or a False appears in hp_warnings_outcome
    no_hp_warnings = (hp_warnings_outcome != True).all(axis=0)
    didnt_record_hp_warnings = (hp_warnings_outcome == None).any(axis=0)
    allowable_hp_idxs = np.where(no_hp_warnings == True)[0]

    assert (
        len(allowable_hp_idxs) > 0
    ), "all of your hp indices resulted in warnings so none is optimal"

    # only work with the ones that you can actually use
    kfold_metrics_outcome = kfold_metrics_outcome[:, allowable_hp_idxs]

    # get extract value for selected criteria
    # from kfold_metrics array for this particular outcome
    def extract_crit(x):
        return x[crit]

    f = np.vectorize(extract_crit)
    vals = f(kfold_metrics_outcome)

    # average across folds
    means = vals.mean(axis=0)

    # get indices of optimal hyperparam(s) - shape: num_hps x num_optimal_hp_settings
    idx_extreme = np.array(np.where(means == finder(means)))

    # warn if hp hit the bounds of your grid search for any hp (there may be >1 hps)
    for ix, this_hp in enumerate(idx_extreme):
        n_hp_vals = means.shape[ix]
        # if there's just one hp parameter, just throw one warning.
        if n_hp_vals == 1:
            log_text(
                "Only one value for hyperparameter number {0} supplied.".format(ix),
                 print_text="warn", #we dont want to suppress this warning
            )
        else:
            # otherwise check if the optimal hp value is on the boundary of the search
            if 0 in this_hp:
                if didnt_record_hp_warnings[allowable_hp_idxs[ix]]:
                    log_text(
                        "The optimal hyperparameter is the lowest of those supplied "
                        + "(it was not checked for precision warnings). "
                        + "hyperparameters supplied. "
                        + "It is index {0} of the orignal hyperparamters passed in. ".format(
                            allowable_hp_idxs[ix]
                        ), print_text=sup
                    )

                else:
                    log_text(
                        "The optimal hyperparameter is the lowest of the acceptable (i.e. no precision warnings) "
                        + "hyperparameters supplied. "
                        + "It is index {0} of the orignal hyperparamters passed in. ".format(
                            allowable_hp_idxs[ix]
                        )
                        + "For reference, {0} of {1} ".format(
                            len(allowable_hp_idxs), len(no_hp_warnings)
                        )
                        + "hyperparamters are considered acceptable; "
                        + "their indices  are {0}.".format(allowable_hp_idxs),
                        print_text = sup
                    )

            if (n_hp_vals - 1) in this_hp:
                if didnt_record_hp_warnings[allowable_hp_idxs[ix]]:
                    log_text(
                        "The optimal hyperparameter is the highest of those supplied "
                        + "(it was not checked for precision warnings). "
                        + "hyperparameters supplied. "
                        + "It is index {0} of the orignal hyperparamters passed in. ".format(
                            allowable_hp_idxs[ix]
                        ), print_text = sup
                    )
                else:
                    log_text(
                        "The optimal hyperparameter is the highest of the acceptable (i.e. no precision warnings) "
                        + "hyperparameters supplied. "
                        + "It is index {0} of the orignal hyperparamters passed in. ".format(
                            allowable_hp_idxs[ix]
                        )
                        + "For reference, {0} of {1} ".format(
                            len(allowable_hp_idxs), len(no_hp_warnings)
                        )
                        + "hyperparameters are considered acceptable; "
                        + "their indices  are {0}.".format(allowable_hp_idxs),
                        print_text = sup
                    )

    # warn if multiple optimal sets of hp
    if idx_extreme.shape[1] > 1:
        log_text( #We don't want to suppress this warning
            "Multiple optimal hyperparameters found for outcome {0}. Indices: {1}".format(
                val, idx_extreme
            ), print_text="warn"
        )

    # select first optimal set
    return tuple(allowable_hp_idxs[idx_extreme[:, 0]])


def get_fold_results_by_hp_idx(kfold_metrics, idxs):
    """Slice model performance metrics array by 
    hyperparameter indices.
    
    Args:
        kfold_metrics (n_folds X n_outcomes X ... ndarray of dict): 
            Model performance array produced by kfold_solve
        idxs (list of tuple): The indices of the hyperparameter values
            swept over in cross-validation. The dimension of the list
            indexes n_outcomes and the dimension of the tuples index ...
            
    Returns:
        n_folds X n_outcomes: Model performance for each fold using the
            set of hyperparameters defined in idxs
    """

    # initialize array of size n_folds X n_outcomes
    res = np.empty(kfold_metrics.shape[:2], dtype=kfold_metrics.dtype)

    for outcome_ix, i in enumerate(idxs):
        # slice this outcome plus the optimal hp's for this outcome
        # (first column is across folds, don't slice)
        slicer = [slice(None), outcome_ix] + list(i)
        res[:, outcome_ix] = kfold_metrics[tuple(slicer)]
    return res


def _get_best_hps(hps, best_idxs):
    best_hps = []
    hp_names = [h[0] for h in hps]
    n_outcomes = 1
    for ox in range(n_outcomes):
        this_best_hps = []
        for hpx, hp in enumerate(best_idxs[ox]):
            this_best_hps.append(hps[hpx][1][hp])
        best_hps.append(this_best_hps)
    hp_names = np.array(hp_names)
    best_hps = np.array(best_hps)
    return best_hps, hp_names
    

def interpret_kfold_results(
    kfold_results, crits, minimize=False, save_weight_path=None, 
    save_weight_fname=None, hps=None
):
    """Return the parsed results of the best performing model from 
    kfold_solve.
    
    Args: 
        kfold_results (dict): As returned by kfold_solve()
        crits (str or list of str): Metric criteria to base contractions
            off of for each dimension. Must be str or list of length n_outcomes
        minimize (bool or list of bool) : Whether to find minimal (True) or maximal
            (False) value of each crit. (e.g. should be False for r2 and True for MSE)
        save_weight_path (optional, str): Path where weights of model should be saved 
            (if not None). Should end in '.npz'. This file will have 3 arrays. 'weights'
            will be n_folds X n_outcomes X n_features. 'hps' will be n_outcomes X n_hyperparams.
            'hp_names' will be n_hyperparams and will have the hyperparemeter names in the same
            order as the values appearing in 'hps'.
        hps (list of 2-tuples): List of hyperparameters tested. Order of the tuples is
            the same as the order they appear in kfold_results. e.g. [('lambda',[0,1,10])].
            Required if save_weight_path is not None so that the optimal HP can be saved with
            the weights.
    Returns: 
        list of tuples: The indices of the best hyperparameters for each outcome. The dimension of
            the list indexes outcomes, the dimension of the tuple indexes hyperparameters.
            If more than one hyperparameter was swept over, this inner dimension will be >1.
            In that case, the order is the same order that was used in the dimensions of 
            kfold_metrics arrays output by the solve function used to generate these results.
        n_folds X n_outcomes 2darray of dict: Model performance array for optimal set of 
            hyperparameters for each outcome across folds
        n_folds X n_outcomes 2darray of 1darray: Model predictions array for optimal set of 
            hyperparameters for each outcome across folds
    """
    kfold_metrics = kfold_results["metrics_test"]
    kfold_preds = kfold_results["y_pred_test"]
    kfold_hp_warnings = kfold_results["hp_warning"]

    if save_weight_path is not None:
        kfold_models = kfold_results["models"]

    kfold_shp = kfold_metrics.shape
    num_folds = kfold_shp[0]
    num_outputs = kfold_shp[1]

    if isinstance(minimize, bool):
        minimize = [minimize for i in range(num_outputs)]
    if isinstance(crits, str):
        crits = [crits for i in range(num_outputs)]

    best_idxs = []
    for j in range(num_outputs):
        this_output_results_by_fold = kfold_metrics.take(indices=j, axis=1)
        this_hp_warnings_by_fold = kfold_hp_warnings.take(indices=j, axis=1)
        best_idxs_for_this_output = find_best_hp_idx(
            this_output_results_by_fold,
            this_hp_warnings_by_fold,
            crits[j],
            minimize=minimize[j],
            val=j,
        )
        best_idxs.append(best_idxs_for_this_output)

    # using the indices of the best hp values, return the model performance across all folds
    # using those hp values
    metrics_best_idx = get_fold_results_by_hp_idx(kfold_metrics, best_idxs)

    # using the indices of the best hp values, return the model predictions across all folds
    # using those hp values
    y_pred_best_idx = get_fold_results_by_hp_idx(kfold_preds, best_idxs)

    # using the indices of the best hp values, return the model weights across all folds
    # using those hp values
    if all([save_weight_path, save_weight_fname]):
        os.makedirs(save_weight_path, exist_ok=True)
        best_hps, hp_names = _get_best_hps(hps, best_idxs)
        models_best_idx = get_fold_results_by_hp_idx(kfold_models, best_idxs)
        np.savez(
            os.path.join(save_weight_path, save_weight_fname),
            weights=models_best_idx, hps=best_hps, hp_names=hp_names
        )

    # return the best idx along with the metrics and preds for all the folds corresponding to that index.
    return best_idxs, metrics_best_idx, y_pred_best_idx


def interpret_single_results(
    kfold_results, crits, minimize=False, save_weight_path=None, hps=None
):
    """Return the parsed results of the best performing model from 
    kfold_solve.
    
    Args: 
        kfold_results (dict): As returned by kfold_solve()
        crits (str or list of str): Metric criteria to base contractions
            off of for each dimension. Must be str or list of length n_outcomes
        minimize (bool or list of bool) : Whether to find minimal (True) or maximal
            (False) value of each crit. (e.g. should be False for r2 and True for MSE)
        save_weight_path (optional, str): Path where weights of model should be saved 
            (if not None). Should end in '.npz'. This file will have 3 arrays. 'weights'
            will be n_features. 'hps' will be n_hyperparams.
            'hp_names' will be n_hyperparams and will have the hyperparemeter names in the same
            order as the values appearing in 'hps'.
        hps (list of 2-tuples): List of hyperparameters tested. Order of the tuples is
            the same as the order they appear in kfold_results. e.g. [('lambda',[0,1,10])].
            Required if save_weight_path is not None so that the optimal HP can be saved with
            the weights.
    Returns: 
        list of tuples: The indices of the best hyperparameters for each outcome. The dimension of
            the list indexes outcomes, the dimension of the tuple indexes hyperparameters.
            If more than one hyperparameter was swept over, this inner dimension will be >1.
            In that case, the order is the same order that was used in the dimensions of 
            kfold_metrics arrays output by the solve function used to generate these results.
        n_folds X n_outcomes 2darray of dict: Model performance array for optimal set of 
            hyperparameters for each outcome across folds
        n_folds X n_outcomes 2darray of 1darray: Model predictions array for optimal set of 
            hyperparameters for each outcome across folds
    """
    kfold_metrics = kfold_results["metrics_test"]
    kfold_preds = kfold_results["y_pred_test"]
    kfold_hp_warnings = kfold_results["hp_warning"]
    if save_weight_path is not None:
        kfold_models = kfold_results["models"]

    kfold_shp = kfold_metrics.shape
    num_folds = kfold_shp[0]
    num_outputs = kfold_shp[1]
    assert num_folds == 1
    if isinstance(minimize, bool):
        minimize = [minimize for i in range(num_outputs)]
    if isinstance(crits, str):
        crits = [crits for i in range(num_outputs)]
    # for each output (column of y), find the best hyperparameter
    # values over all folds
    best_idxs = []
    for j in range(num_outputs):
        this_output_results_by_fold = kfold_metrics.take(indices=j, axis=1)
        this_hp_warnings_by_fold = kfold_hp_warnings.take(indices=j, axis=1)
        best_idxs_for_this_output = find_best_hp_idx(
            this_output_results_by_fold,
            this_hp_warnings_by_fold,
            crits[j],
            minimize=minimize[j],
            val=j,
        )
        best_idxs.append(best_idxs_for_this_output)

    # using the indices of the best hp values, return the model performance across all folds
    # using those hp values
    metrics_best_idx = get_fold_results_by_hp_idx(kfold_metrics, best_idxs)

    # using the indices of the best hp values, return the model predictions across all folds
    # using those hp values
    y_pred_best_idx = get_fold_results_by_hp_idx(kfold_preds, best_idxs)

    # using the indices of the best hp values, return the model weights across all folds
    # using those hp values
    if save_weight_path is not None:
        best_hps, hp_names = _get_best_hps(hps, best_idxs)
        models_best_idx = get_fold_results_by_hp_idx(kfold_models, best_idxs)
        np.savez(
            save_weight_path,
            weights=models_best_idx[0][0],
            hps=best_hps[0],
            hp_names=hp_names,
        )
        os.chmod(save_weight_path, 0o774)

    # return the best idx along with the metrics and preds for all the folds corresponding to that index.
    # fold_idx = 0
    # column_idx = 0
    if num_outputs == 1:
        return best_idxs[0], metrics_best_idx[0][0], y_pred_best_idx[0][0]
    else:
        return best_idxs, metrics_best_idx[0], y_pred_best_idx[0]

    
    
def get_avg_r2s_from_kfold_results_for_each_lambda(kfold_results, lambdas, metric="r2_score", minimize=False):
    """
    This function is unique in that is able to interpret kfold_results objects that are returned from the hurdle model. 
    
    These are structurally different from the MOSAIKS outputs. 
    
    Returns:
    -----
        ma.array : masked array that is the length of the number of lambdas used in the model.
            When there is an hp warning, those values are masked.
    
    
    """
    if minimize:
        finder = np.min
    else:
        finder = np.max
        
    kfold_metrics = kfold_results["metrics_test"]
    num_folds = kfold_metrics.shape[0]
    num_models = kfold_metrics.shape[2]
    num_lambdas = len(lambdas)

    if num_models != num_lambdas:
        hurdle = True
        num_thresholds = int(num_models/num_lambdas)
        log_text("Shape of kfold results suggests that we are runnning a hurdle model")
    else:
        num_thresholds=1

    start_idx = 0
    end_idx = num_thresholds


    avg_lam_r2 = []
    any_lam_hp_warning = []

    for i in range(num_lambdas):

        fold_r2s = []
        fold_hp_warning = []

        for j in range(num_folds):
            lambda_results = kfold_metrics[j][0][start_idx:end_idx]
            best_thresh_perf = finder([k[metric] for k in lambda_results])


            fold_r2s.append(best_thresh_perf)
            fold_hp_warning.append(kfold_results["hp_warning"][j][0][i])

        start_idx += num_thresholds
        end_idx += num_thresholds

        avg_lam_r2.append(np.mean(fold_r2s))
        any_lam_hp_warning.append(any(fold_hp_warning))

    return np.ma.array(avg_lam_r2, mask=any_lam_hp_warning)


def get_kfold_results_from_main(app):

    main_contents = os.listdir(c.main_pred_dir)
    matching_items = [app in item for item in main_contents]
    matching_idxs = [i for i, item in enumerate(matching_items) if item]

    if len(matching_idxs) > 1:
        log_text( ("Multiple matching applications found in main pred directory. "
        "Using shortest file length option..."), print_text="warn")
        log_text("app = "+ app)
        log_text("Matching objects = " + str(np.array(main_contents)[np.array(matching_idxs)]))

        idx = np.argmin([len(main_contents[item]) for item in matching_items])
        matching_idxs = [idx]
        
        log_text("Selected object = " + str(main_contents[matching_idxs[0]]) )
        
    elif len(matching_idxs) == 0:
        log_text( ("No matching applications found in main pred directory. "
        "Cannot automatically get kfold_results object from main pred directory"))
        raise Exception

    
    kfold_results = pd.read_pickle( os.path.join(c.main_pred_dir, main_contents[matching_idxs[0]]))
        
    return kfold_results


def get_config_from_filename(app, c=c):
    '''
    Return a runable config using a combination of defaults and the filename of
    the best pred version of the app
    '''
    fname = glob.glob(c.main_pred_dir + "/" + app + "_[rhc]*")

    assert len(fname) <= 1, "Need only 1 pred per app"
    if len(fname) == 0:
        print('No result found for {}'.format(app))
        return
    fname = fname[0]
    
    print('Populating config from {}'.format(fname))
    
    # start with the default config, then modify based on filename
    c_app = parse_metadata(app)

    # solve function
    if re.search('_class_', fname):
        c_app['solve_function'] = 'OVR_classifier'
    elif re.search('_hurdle_', fname):
        c_app['solve_function'] = 'Hurdle'
    elif re.search('_reg_', fname):
        c_app['solve_function'] = 'Ridge'
    else:
        raise KeyError("Solve function not found")
    
    # logs and levels
    if re.search('_log_', fname):
        c_app['tranformation'] = 'log'

    # set bounds
    
    if re.search('_observedbounds_', fname):
        c_app['bounds_pred'] = [None, None]
        c_app['bound_replace_None_w_observed'] = True
    elif re.search('_nobound_', fname):
        c_app['bounds_pred'] = [None, None]
        c_app['bound_replace_None_w_observed'] = False
    elif re.search('_combobound_', fname):
        # note that no best perfoming model uses combo bounds
        raise NotImplementedError('Not implemented for combo bounds')
    # intuitive bounds given by the default c_app
    
    if re.search('_nointercept_', fname):
        c_app['intercept'] = False
    else:
        c_app['intercept'] = True
        
    # continent model?
    if re.search('_continent_', fname):
        c_app['region_type'] = 'continent'
    elif re.search('_country_', fname):
        c_app['region_type'] = 'country'
    else:
        c_app['region_type'] = None

    return c_app