import gc
import glob
import itertools
import os
import pickle
import shutil
import time
import warnings

import numpy as np

from functools import reduce
import sklearn.metrics as metrics
from mosaiks import config as c
from mosaiks.utils.logging import log_text
from mosaiks.solve import data_parser as parse
from mosaiks.solve import interpret_results as ir
from mosaiks.utils import io as mio
from scipy.linalg.misc import LinAlgWarning
from sklearn.linear_model._base import _preprocess_data
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.exceptions import ConvergenceWarning



DEBUG = False

if c.GPU:
    import torch
    c.GPU = torch.cuda.is_available()

if c.GPU:
    import cupy as xp
    from cupy import linalg

    linalg_solve_kwargs = {}
    asnumpy = xp.asnumpy
    mempool = xp.get_default_memory_pool()
    pinned_mempool = xp.get_default_pinned_memory_pool()
else:
    from scipy import linalg

    linalg_solve_kwargs = {"sym_pos": True}
    xp = np
    asnumpy = np.asarray

    
def ridge_regression(
    X_train,
    X_test,
    y_train,
    y_test,
    svd_solve=False,
    lambdas=[1e2],
    return_preds=True,
    return_model=False,
    clip_bounds=None,
    intercept=False,
    allow_linalg_warning_instances=False
):
    """Train ridge regression model for a series of regularization parameters.
    Optionally clip the predictions to bounds. Used as the default solve_function
    argument for single_solve() and kfold_solve() below.

    Parameters
    ----------
        X_{train,test} : :class:`numpy.ndarray`
            Features for training/test data (n_obs_{train,test} X n_ftrs 2darray).
        y_{train,test} : :class:`numpy.ndarray`
            Labels for training/test data (n_obs_{train,test} X n_outcomes 2darray).
        svd_solve : bool, optional
            If true, uses SVD to compute w^*, otherwise does matrix inverse for each
            lambda.
        lambdas : list of floats, optional
            Regularization values to sweep over.
        return_preds : bool, optional
            Whether to return predictions for training and test sets.
        return_model : bool, optional
            Whether to return the trained weights that define the ridge regression
            model.
        clip_bounds : array-like, optional
            If None, do not clip predictions. If not None, must be ann array of
            dimension ``n_outcomes X 2``. If any of the elements of the array are None,
            ignore that bound (e.g. if a row of the array is [None, 10], apply an upper
            bound of 10 but no lower bound).
        intercept : bool, optional
            Whether to add an unregulated intercept (or, equivalently, center the X and
            Y data).
        allow_linalg_warning_instances : bool, optional
            If False (default), track for which hyperparameters did ``scipy.linalg`` 
            raise an ill-conditioned matrix error, which could lead to poor performance.
            This is used to discard these models in a cross-validation context. If True,
            allow these models to be included in the hyperparameter grid search. Note
            that these errors will not occur when using ``cupy.linalg`` (i.e. if a GPU 
            is detected), so the default setting may give differing results across 
            platforms.

    Returns
    -------
    dict of :class:`numpy.ndarray`
        The results dictionary will always include the following key/value pairs:
            ``metrics_{test,train}`` : array of dimension n_outcomes X n_lambdas
                Each element is a dictionary of {Out-of,In}-sample model performance 
                metrics for each lambda

        If ``return_preds``, the following arrays will be appended in order:
            ``y_pred_{test,train}`` : array of dimension n_outcomes X n_lambdas
                Each element is itself a 1darray of {Out-of,In}-sample predictions for 
                each lambda. Each 1darray contains n_obs_{test,train} values

        if return_model, the following array will be appended:
            ``models`` : array of dimension n_outcomes X n_lambdas:
                Each element is itself a 1darray of model weights for each lambda. Each 
                1darray contains n_ftrs values
    """
    # get dimensions needed to shape arrays
    n_ftrs, n_outcomes, n_obs_train, n_obs_test = get_dim_lengths(
        X_train, y_train, y_test
    )
    n_lambdas = len(lambdas)

    # center data if needed
    X_train, y_train, X_offset, y_offset, _ = _preprocess_data(
        X_train, y_train, intercept, normalize=False
    )

    # set up the data structures for reporting results
    results_dict = _initialize_results_arrays(
        (n_outcomes, n_lambdas), return_preds, return_model
    )

    t1 = time.time()

    # send to GPU if available
    X_train = xp.asarray(X_train)
    y_train = xp.asarray(y_train)

    if DEBUG:
        if GPU:
            log_text(f"Time to transfer X_train and y_train to GPU: {time.time() - t1}")
        t1 = time.time()

    # precomputing large matrices to avoid redundant computation
    if svd_solve:
        # precompute the SVD
        U, s, Vh = linalg.svd(X_train, full_matrices=False)
        V = Vh.T
        UT_dot_y_train = U.T.dot(y_train)
    else:
        XtX = X_train.T.dot(X_train)
        XtY = X_train.T.dot(y_train)
        
    if DEBUG:
        t2 = time.time()
        log_text("Time to create XtX matrix: {}".format(t2 - t1))
        

    # iterate over the lambda regularization values
    training_time = 0
    pred_time = 0
    for lx, lambdan in enumerate(lambdas):
        if DEBUG:
            t3 = time.time()
        
        # train model
        if svd_solve:
            s_lambda = s / (s ** 2 + lambdan * xp.ones_like(s))
            model = (V * s_lambda).dot(UT_dot_y_train)
            lambda_warning = None
        else:
            with warnings.catch_warnings(record=True) as w:
                # bind warnings to the value of w
                warnings.simplefilter("always")
                lambda_warning = False
                model = linalg.solve(
                    XtX + lambdan * xp.eye(n_ftrs, dtype=np.float64),
                    XtY,
                    **linalg_solve_kwargs,
                )

                # if there is a warning
                if len(w) > 1:
                    for this_w in w:
                        log_text(this_w.message)
                    # more than one warning is bad
                    raise Exception("warning/exception other than LinAlgWarning")
                if len(w) > 0:
                    # if it is a linalg warning
                    if w[0].category == LinAlgWarning:
                        log_text("linalg warning on lambda={0}: ".format(lambdan))
                        # linalg warning
                        if not allow_linalg_warning_instances:
                            log_text("we will discard this model upon model selection")
                            lambda_warning = True
                        else:
                            lambda_warning = None
                            log_text("we will allow this model upon model selection")
                    else:
                        raise Exception("warning/exception other than LinAlgWarning")
        
        if DEBUG:
            t4 = time.time()
            training_time += (t4 - t3)
            log_text(f"Training time for lambda {lambdan}: {t4 - t3}")

        #####################
        # compute predictions
        #####################

        
        # send to gpu if available
        X_test = xp.asarray(X_test)
        y_test = xp.asarray(y_test)
        y_offset = xp.asarray(y_offset)
        X_offset = xp.asarray(X_offset)

        if DEBUG: t5 = time.time()
        
        # train
        pred_train = X_train.dot(model) + y_offset
        pred_train = y_to_matrix(pred_train)

        # test
        intercept_term = y_offset - X_offset.dot(model)
        pred_test = X_test.dot(model) + intercept_term
        pred_test = y_to_matrix(pred_test)
        
        # clip if needed
        if clip_bounds is not None:
            for ix, i in enumerate(clip_bounds):
                # only apply if both bounds aren't None for this outcome
                if not (i == None).all():
                    pred_train[:, ix] = xp.clip(pred_train[:, ix], *i)
                    pred_test[:, ix] = xp.clip(pred_test[:, ix], *i)
                    
        if DEBUG:
            t6 = time.time()
            pred_time += (t6-t5)

        # bring back to cpu if needed
        pred_train, pred_test = asnumpy(pred_train), asnumpy(pred_test)
        y_train, y_test, model, intercept_term = (
            y_to_matrix(asnumpy(y_train)),
            y_to_matrix(asnumpy(y_test)),
            y_to_matrix(asnumpy(model)),
            y_to_matrix(asnumpy(intercept_term))
        )

        # create tuple of lambda index to match argument structure
        # of _fill_results_arrays function
        hp_tuple = (lx,)

        # Transpose model results so that n_outcomes is first dimension
        # so that _fill_results_array can handle it
        model = model.T

        # populate results dict with results from this lambda
        results_dict = _fill_results_arrays(
            y_train,
            y_test,
            pred_train,
            pred_test,
            model,
            intercept_term,
            hp_tuple,
            results_dict,
            hp_warning=lambda_warning,
        )
    if DEBUG:
        log_text("Training time: {}".format(training_time))
        log_text("Prediction time: {}".format(pred_time))
        log_text("Total time: {}".format(time.time() - t1))
    return results_dict


def kfold_solve(
    X,
    y,
    locations,
    solve_function=ridge_regression,
    num_folds=5,
    return_preds=True,
    return_model=True,
    fit_model_after_tuning = False,
    **kwargs_solve,
):
    """A general skeleton function for computing k-fold cross validation solves.

    Args:
        X (n_obs X n_ftrs 2darray): Feature matrix
        y (n_obs X n_outcomes 2darray): Attribute matrix
        locations (n_obs x 1 for polygons or n_obs x 2 for lon, lat locations on tiles): array
        solve_function (func): Which solve function in this module will you be using
        num_folds (int): How many folds to use for CV
        return_preds (bool): Return predictions for training and test sets?
        return_model (bool): Return the trained weights that define the ridge regression
            model?
        fit_model_after_tuning (bool): Option to fit a final model with all training data and the 
            optimal hyperparameter
        kwargs_solve (dict): Parameters to pass to the solve func

    Returns:
        Dict of ndarrays.
            The dict will always start with the following 4 key:value pairs. "..."
                refers to a number of dimensions equivalent to the number of
                hyperparameters, where each dimension has a length equal to the number
                of values being tested for that hyperparameter. The number of
                hyperparameters and order returned is defined in the definition of the
                particular solve function we have passed as the solve_function argument:
                    metrics_test: n_folds X n_outcomes X ... ndarray of dict:
                        Out-of-sample model performance metrics for each fold, for each
                        outcome, for each hyperparameter value
                    metrics_train: n_folds X n_outcomes X ... ndarray of dict: In-sample
                        model performance metrics
                    obs_test: n_folds X  n_outcomes  X ... array of ndarray of float64:
                        Out-of-sample observed values for each fold
                    obs_train: n_folds X  n_outcomes X ... array of ndarray of float64:
                        In-sample observed values
                    cv: :py:class:`sklearn.model_selection.KFold` : kfold
                        cross-validation splitting object used

            If return_preds, the following arrays will included:
                preds_test: n_folds X  n_outcomes X ... ndarray of ndarray of float64:
                    Out-of-sample predictions or each fold, for each outcome, for each
                    hyperparameter value
                preds_train: n_folds X n_outcomes X ... ndarray of ndarray of float64:
                    In-sample predictions

            if return_model, the following array will be included:
                models: n_folds X n_outcomes X ... ndarray of same type as model: Model
                    weights/parameters. xxx here is of arbitrary dimension specific to
                    solve_function
    """
    assert num_folds > 1

    kf = KFold(n_splits=num_folds, shuffle=True, random_state=0)
    

    # keep track of all runs over several iterations
    kfold_metrics_test = []
    kfold_metrics_train = []
    kfold_preds_test = []
    kfold_preds_train = []
    kfold_y_train = []
    kfold_y_test = []
    locations_test = []
    kfold_models = []
    kfold_intercepts = []
    hp_warnings = []
    i = -1
    
    if (solve_function is hurdle_model_2) or (solve_function is new_hurdle_model_2):
        step1 = kwargs_solve.pop("step1")
    
    elif "zero_rep" in kwargs_solve:
        _ = kwargs_solve.pop("zero_rep")
    
    # ensure that thresh parameter cannot be passed to solve functions that do not accept it
    if (solve_function is ridge_regression) or (solve_function is OVR_classifier):
        if "thresh" in kwargs_solve:
            _ = kwargs_solve.pop("thresh")


    for train_idxs, val_idxs in kf.split(X):
        i += 1
        log_text("on fold ({0} of {1}): ".format(i+1, num_folds))
        
        X_train, X_val = X[train_idxs], X[val_idxs]
        y_train, y_val = y[train_idxs], y[val_idxs]
        
        locations_val = locations[val_idxs]
        locations_test.append(locations_val)
        
        if (solve_function is hurdle_model_2) or (solve_function is new_hurdle_model_2):
            kwargs_solve["step1_train"] = step1['y_pred_train'][i]
            kwargs_solve["step1_test"] = step1['y_pred_test'][i]

        # record train/test obs for this split
        kfold_y_train.append(y_train)
        kfold_y_test.append(y_val)

        # call solve func
        solve_results = solve_function(
            X_train,
            X_val,
            y_train,
            y_val,
            return_preds=return_preds,
            return_model=return_model,
            **kwargs_solve,
        )

        # record performance metrics
        kfold_metrics_test.append(solve_results["metrics_test"])
        kfold_metrics_train.append(solve_results["metrics_train"])

        # record optional preds and model parameters
        if return_preds:
            kfold_preds_test.append(solve_results["y_pred_test"])
            kfold_preds_train.append(solve_results["y_pred_train"])
        
        if return_model:
            kfold_models.append(solve_results["models"])
            kfold_intercepts.append(solve_results["intercept"])

        # recpord np warnings
        hp_warnings.append(solve_results["hp_warning"])

    # Return results
    rets = {
        "metrics_test": np.array(kfold_metrics_test,dtype=object),
        "metrics_train": np.array(kfold_metrics_train,dtype=object),
        "y_true_test": np.array(kfold_y_test,dtype=object),
        "y_true_train": np.array(kfold_y_train,dtype=object),
        "locations_test": np.array(locations_test,dtype=object),
        "hp_warning": np.array(hp_warnings,dtype=object),
        "cv": kf,
    }

    if return_preds:
        rets["y_pred_test"] = np.array(kfold_preds_test,dtype=object)
        rets["y_pred_train"] = np.array(kfold_preds_train,dtype=object)

    if return_model:
        rets["models"] = np.array(kfold_models,dtype=object)
        rets["intercepts"] = np.array(kfold_intercepts,dtype=object)
    
    if fit_model_after_tuning:
        if solve_function is not ridge_regression:
            log_text( ("Fit model after tuning has only been "
             "implemented with the ridge solve function"
             " no final model will be fit"), print_text="warn")
        else:
            best_lambda_idx = ir.interpret_kfold_results(rets, "r2_score")[0][0][0]
            best_lambda = kwargs_solve["lambdas"][best_lambda_idx]
            intercept = kwargs_solve.get("intercept", False)

            (rets["prediction_model_weights"], 
            rets["prediction_model_intercept"]) = custom_ridge(X,y, lam=best_lambda, intercept=intercept)

    return rets

def kfoldsolve_tunelambda( 
    X,
    y,
    locations,
    solve_function=ridge_regression,
    num_folds=5,
    generate_lambda_default=True,
    init_log_min_lambda= -4,
    init_log_max_lambda = 1,
    n_lambdas = 5,
    max_iter = 5,
    return_preds=True,
    return_model=True,
    fit_model_after_tuning = True,
    run_fine_tuning_step = False,
    **kwargs_solve,
):
    """A general function for computing k-fold cross validation solves while tuning lambda values.
    
    Follows these steps:
    - Starts with a list of lambdas, either passed in from kwargs_solve or generated by this function
    - If generated by this function, the default starts with [n] values from e^[min] to e^[max] (as default) (scaled by mean y if y is continuous).
    - Calls kfold_solve and ir.lambda_at_min_max_of_acceptable to find the best of the initial values
    - If the best value is either the min or the max of the range attempted, extends the range in logspace until the best value is not the min or max
    - (Exception: If at low end, lambda will not extend below zero)
    - (optional final step) Once the best value is found, tries a narrower range of [n] values, centered on the best value from the previous round.
    - Returns kfold_results from the final linear space lambda search

    Args:
    
        Same as kfoldsolve above, except also includes:
        
            generate_lambda_default: Bool - True if you want to generate default values; 
                        False if you want to use lambdas passed in from config.
                        When true, sets initial lambda values to e^min to e^max
                        (passed in from lambda_tuners), scaled by the mean of y (when y is quantitative)

            init_log_min_lambda: int - exp of min value of lambda to sweep over (i.e., min lambda will be 10^x), 
            init_log_max_lambda: int - exp of max value of lambda to sweep over (i.e., max lambda will be 10^x), 
            n_lambdas: int - number of lambdas to sweep over. All steps use same number of lambdas
            max_iter: int - maximum times to try to get a non-zero value for best_lambda_idx
            run_fine_tuning_step: Bool - Indicates whether you want to run an extra model for the fine tuning of lambdas 

    """
    
    # Obtain and scale default values for lambdas (if lambda_default = True)
    if generate_lambda_default==True:
        if "lambdas" in kwargs_solve.keys():
            warning_message = ("Warning: Unclear behavior. \n"
                               "generate_lambda_default is set to True and kwargs_solve also contains `lambda` key. "
                               "Lambda values passed in via kwargs_solve will be ignored.")
            log_text(warning_message, print_text='warn')
            
        lambda_init = np.logspace(init_log_min_lambda,init_log_max_lambda,n_lambdas)
        
        if np.issubdtype(y.dtype, np.number): # Check to see if Y is a numeric numpy object. If so, scale lambdas.
            lambda_scale = np.absolute(np.mean(y)) + 1
            lambda_init  = lambda_init * lambda_scale
            
        kwargs_solve['lambdas'] = lambda_init
    else:
        pass
    
    ### FIRST STEP
    # Run solve using an initial set of lambdas, passed into function or set above. 
    # We force the while loop to run using the initial conditions below

    minimum = False #initializing conditions to begin the while loop. Ensures minimal repitition of code
    maximum = False

    i = 0
    
    while minimum or maximum or (i == 0):
        log_text(f"while loop #: {i}")

        if minimum:
            low_lambda = lambda_optimal * .001 # try a min value two orders of magnitude lower than previous optimal
            high_lambda = lambda_optimal * .1  # max will be previously optimal lambda
        
        elif maximum:
            low_lambda = lambda_optimal * 10 # min will be previously optimal lambda
            high_lambda = lambda_optimal * 1000 # try a max value two orders of magnitude higher than previous optimal
        
        if i != 0: # in the first step, we pass in an initial set of lambdas
            prev_best_r2 = best_r2
            prev_rets = rets.copy()
            prev_lambdas = kwargs_solve["lambdas"]
            prev_lambda_optimal = lambda_optimal
            prev_best_lambda_idx = best_lambda_idx

            kwargs_solve['lambdas'] = np.logspace(np.log10(low_lambda), np.log10(high_lambda), 3) #hard code fewer lams in while loop refine 
        
        
        log_text("running kfold_solve with lambdas:")
        log_text(str(np.round(kwargs_solve["lambdas"],6)))
        rets = kfold_solve(
                X,
                y,
                locations,
                solve_function=solve_function,
                num_folds=num_folds,
                return_preds=True,
                return_model=return_model,
                **kwargs_solve)
     
        r2s = ir.get_avg_r2s_from_kfold_results_for_each_lambda(rets, kwargs_solve["lambdas"])
        best_r2 = np.max(r2s)
        best_lambda_idx = np.argmax(r2s)
        lambda_optimal = kwargs_solve["lambdas"][best_lambda_idx]
        r2s_dropped_hp_warnings = np.array(r2s)[~r2s.mask] #here we make another array with dropped hp warnings; these have different indices
        all_hp_warnings = len(r2s_dropped_hp_warnings) == 0 # bool to indicate if we have hp warnings on all lambdas

        log_text("r2s:")
        log_text(str(np.round(r2s,4)))
        log_text("lambda optimal: " + str(lambda_optimal))

        # Not on initial sweep and prev iteration has best r2 or current iteration has all hp warnings
        if i!=0 and ( (prev_best_r2 > best_r2) or (all_hp_warnings)):
            log_text("Previous set of lambdas had better perf than most recent run")
            if all_hp_warnings:
                log_text("Current run has hp warnings on all lambdas")
            rets = prev_rets
            best_r2 = prev_best_r2
            best_lambda_idx = prev_best_lambda_idx
            kwargs_solve["lambdas"] = prev_lambdas
            lambda_optimal = prev_lambda_optimal
            break


    ### CONDITIONAL SECOND STEP
    ## When initial lambdas have an optimal at the edge of the range, expand the range
    ## This while loop only when the optimal lambda from the previous sweep was the min or max of acceptable values

        if all_hp_warnings:
            raise Exception("hp warnings on all initial lambdas; cannot autotune")

        elif len(r2s_dropped_hp_warnings) == 1:
            log_text("hp warnings on almost all input lambdas; we will proceed with the only acceptable lambdas", 
            print_text="warn")
            break

        elif np.argmax(r2s_dropped_hp_warnings) == 0:
            log_text("lambda at min of acceptable range")
            minimum = True
            
        elif np.argmax(r2s_dropped_hp_warnings) == (len(r2s_dropped_hp_warnings)-1):
            log_text("lambda at max of acceptable range")
            maximum = True
        else:
            log_text("optimal lambda not at extrema, breaking while loop")
            break
        
        if minimum and maximum:
            log_text("continuing while loop would be circular, breaking loop")
            break
        
        if i != 0:
            improvement_delta = best_r2 - prev_best_r2
            log_text(f"improvement delta: " + str(improvement_delta))
        
            if (improvement_delta < .001):
                log_text("improvement delta < .001, breaking while loop")
                break

        if i >= max_iter:
            warning_message = ("Warning: Reached maximum attempts at finding optimal lambda."
                               " This behavior is not expected and highly unusual. Check input data and results.")
            log_text(warning_message, print_text='warn')
            break
            
        i += 1 
        ## End of while loop

    if not run_fine_tuning_step:
        log_text("Not running final, fine tune stage in autotune lambda")
        rets['lambdas'] = np.array(kwargs_solve['lambdas'])

        if fit_model_after_tuning:
            if solve_function is not ridge_regression:
                log_text( ("Fit model after tuning has only been "
                "implemented with the ridge solve function"
                " no final model will be fit"), print_text="warn")
            else:
                intercept = kwargs_solve.get("intercept", False)
                (rets["prediction_model_weights"], 
                rets["prediction_model_intercept"]) = custom_ridge(X,y, 
                lam=lambda_optimal, intercept=intercept)
        return rets
    
    ### FINAL STEP
    # After obtaining a best lambda that is not the min or max of the tested values, fine-tune it further
    # by testing a smaller range of lambdas from the lambda below the best to the lambda above the best
    low_lambda = lambda_optimal * 0.5 # one half order of magnitude lower than previous best
    high_lambda = lambda_optimal * 5 # one half order of magnitude higher than previous best
    
    kwargs_solve['lambdas'] = np.logspace(np.log10(low_lambda), np.log10(high_lambda),n_lambdas)
    
    # generate kfold_results for the second, more refined set of lambdas
    log_text("running final kfold_solve with lambdas:")
    log_text(str(kwargs_solve["lambdas"]))
    
    rets = kfold_solve(
            X,
            y,
            locations,
            solve_function=solve_function,
            num_folds=num_folds,
            return_preds=return_preds,
            return_model=return_model,
            **kwargs_solve)

    rets['lambdas'] = np.array(kwargs_solve['lambdas']) # add second, refined set of tuning lambda values to kfold_results
    if fit_model_after_tuning:
        if solve_function is not ridge_regression:
            log_text( ("Fit model after tuning has only been "
            "implemented with the ridge solve function"
            " no final model will be fit"), print_text="warn")
        else:
            r2s = ir.get_avg_r2s_from_kfold_results_for_each_lambda(rets, kwargs_solve["lambdas"])
            best_r2 = np.max(r2s)
            best_lambda_idx = np.argmax(r2s)
            lambda_optimal = kwargs_solve["lambdas"][best_lambda_idx]
            
            intercept = kwargs_solve.get("intercept", False)
            (rets["prediction_model_weights"], 
            rets["prediction_model_intercept"]) = custom_ridge(X,y, 
            lam=lambda_optimal, intercept=intercept)

    return rets


def single_solve(
    X_train,
    X_val,
    y_train,
    y_val,
    solve_function=ridge_regression,
    return_preds=True,
    return_model=False,
    **kwargs_solve,
):
    """A general skeleton function for computing k-fold cross validation solves.

    Args:
        X_train, X_val (n_train_obs X n_ftrs 2darray), (n_test_obs X n_ftrs 2darray): Feature matrices
        y_train, y_val: y (n_train_obs X n_outcomes 2darray), (n_test_obs X n_outcomes 2darray) : Attribute matrices
        solve_function (func): Which solve function in this module will you be using
        num_folds (int): How many folds to use for CV
        return_preds (bool): Return predictions for training and test sets?
        return_model (bool): Return the trained weights that define the ridge regression model?
        kwargs_solve (dict): Parameters to pass to the solve func

    Returns:
        Dict of ndarrays.
            The dict will always start with the following 4 key:value pairs. "..." refers to a number
            of dimensions equivalent to the number of hyperparameters, where each dimension
            has a length equal to the number of values being tested for that hyperparameter.
            The number of hyperparameters and order returned is defined in the definition of
            the particular solve function we have passed as the solve_function argument:
                metrics_test:  n_outcomes X ... ndarray of dict: Out-of-sample model performance
                    metrics for each fold, for each outcome, for each hyperparameter value
                metrics_train: n_outcomes X ... ndarray of dict: In-sample model performance metrics
                obs_test: n_folds X  n_outcomes  X ... array of ndarray of float64: Out-of-sample observed values
                    for each fold
                obs_train:  n_outcomes X ... array of ndarray of float64: In-sample observed values
                cv: :py:class:`sklearn.model_selection.KFold` : kfold cross-validation splitting object used

            If return_preds, the following arrays will included:
                preds_test:  n_outcomes X ... ndarray of ndarray of float64: Out-of-sample predictions
                    for each fold, for each outcome, for each hyperparameter value
                preds_train: n_outcomes X ... ndarray of ndarray of float64: In-sample predictions

            if return_model, the following array will be included:
                models: n_outcomes X ... ndarray of same type as model: Model weights/parameters. xxx here is of
                    arbitrary dimension specific to solve_function
    """
    # call solve func
    solve_results = solve_function(
        X_train,
        X_val,
        y_train,
        y_val,
        return_preds=return_preds,
        return_model=return_model,
        **kwargs_solve,
    )

    # Return results wrapped to interface with interpret_results functoins
    rets = {
        "metrics_test": np.array([solve_results["metrics_test"]]),
        "metrics_train": np.array([solve_results["metrics_train"]]),
        "y_true_test": np.array(y_val),
        "y_true_train": np.array(y_train),
        "hp_warning": np.array([solve_results["hp_warning"]]),
    }

    if return_preds:
        rets["y_pred_test"] = np.array([solve_results["y_pred_test"]])
        rets["y_pred_train"] = np.array([solve_results["y_pred_train"]])

    if return_model:
        rets["models"] = np.array([solve_results["models"]])

    if GPU:
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()
    return rets


def compute_metrics(true, pred):
    """ takes in a vector of true values, a vector of predicted values. To add more metrics, 
    just add to the dictionary (possibly with a flag or when
    it is appropriate to add) """
    res = dict()

    residuals = true - pred
    res["mse"] = np.sum(residuals ** 2) / residuals.shape[0]
    res["r2_score"] = metrics.r2_score(true, pred)

    return res


def _initialize_results_arrays(arr_shapes, return_preds, return_models):
    # these must be instantiated independently
    results_dict = {
        "metrics_test": np.empty(arr_shapes, dtype=dict),
        "metrics_train": np.empty(arr_shapes, dtype=dict),
    }
    if return_preds:
        results_dict["y_pred_test"] = np.empty(arr_shapes, dtype=np.ndarray)
        results_dict["y_pred_train"] = np.empty(arr_shapes, dtype=np.ndarray)
    if return_models:
        results_dict["models"] = np.empty(arr_shapes, dtype=np.ndarray)
        results_dict["intercept"] = np.empty(arr_shapes, dtype=np.ndarray)

    # for numerical precision tracking
    results_dict["hp_warning"] = np.empty(arr_shapes, dtype=object)
    results_dict["hp_warning"].fill(None)
    return results_dict


def _fill_results_arrays(
    y_train,
    y_test,
    pred_train,
    pred_test,
    model,
    intercept,
    hp_tuple,
    results_dict,
    hp_warning=None,
):
    """Fill a dictionary of results with the results for this particular
    set of hyperparameters.

    Args:
        y_{train,test} (n_obs_{train,test} X n_outcomes 2darray of float)
        pred_{train,test} (n_obs_{train,test} X n_outcomes 2darray of float)
        model (n_outcomes 1darray of arbitrary dtype)
        hp_tuple (tuple): tuple of hyperparameter values used in this model
        results_dict (dict): As created in solve functions, to be filled in.
    """

    n_outcomes = y_train.shape[1]
    for i in range(n_outcomes):

        # get index of arrays that we want to fill
        # first dimension is outcome, rest are hyperparams
        this_ix = (i,) + hp_tuple

        # compute and save metrics
        results_dict["metrics_train"][this_ix] = compute_metrics(
            y_train[:, i], pred_train[:, i]
        )
        results_dict["metrics_test"][this_ix] = compute_metrics(
            y_test[:, i], pred_test[:, i]
        )

        # save predictions if requested
        if "y_pred_test" in results_dict.keys():
            results_dict["y_pred_train"][this_ix] = pred_train[:, i]
            results_dict["y_pred_test"][this_ix] = pred_test[:, i]

        # save model results if requested
        if "models" in results_dict.keys():
            results_dict["models"][this_ix] = model[i]
            results_dict["intercept"][this_ix] = intercept

        # save hp warnings if thats desired
        results_dict["hp_warning"][this_ix] = hp_warning

    return results_dict


def y_to_matrix(y):
    """ ensures that the y value is of non-empty dimesnion 1 """
    if (type(y) == list) or (type(y)==str) or (type(y)==float):
        y = np.array(y)
        
    input_shp = y.shape
    if len(input_shp) == 0:
        return y
    if (len(input_shp) == 1) or (input_shp[0]==1):
        y = y.reshape(-1, 1)
    return y


def get_dim_lengths(X_train, Y_train, Y_test=None):
    """ packages data dimensions into one object"""
    if Y_train.ndim == 1:
        n_outcomes = 1
    else:
        n_outcomes = Y_train.shape[1]
    n_ftrs = X_train.shape[1]
    n_obs_trn = Y_train.shape[0]

    results = [n_ftrs, n_outcomes, n_obs_trn]
    if Y_test is not None:
        results.append(Y_test.shape[0])
    return results


def split_world_sample_solve(
    X,
    Y,
    latlonsdf,
    sample=0,
    subset_n=slice(None),
    subset_feat=slice(None),
    num_folds=5,
    solve_function=ridge_regression,
    globalclipping=False,
    **kwargs_solve,
):
    """
    runs a cross-validated solve on a subset of X, Y data defined by the sampling indicator contained in
    the latlonsdf object. 
    
    input:
        X, Y are the features and labels matrices, respectively.
        latlonsdf is a pandas dataframe of lat-lon combinations, containing a column called 'samp' which
           contains an indicator of which sample each lat-lon falls into. 
        sample is a scalar from 0 to 5 indicating which subregion of the world you want to solve for.   
        subset_n and subset_feat can be used to subset observations (_n) and.or features (_feat)
        num_folds, solve_function are as described in kfold_solve
        globalclipping is logical; True implies clipping across the whole distribution in Y, False 
            implies clipping within each sample passed into the function.
    
    returns: 
        kfold_results object from the function solve.kfold_solve()
    """

    # limit to just your sample
    ids_samp = np.where(latlonsdf["samp"] == sample)
    X_samp = X.iloc[ids_samp]
    Y_samp = Y.iloc[ids_samp]
    latlonsdf_samp = latlonsdf.iloc[ids_samp]

    # latlons back to ndarray
    this_latlons_samp = latlonsdf_samp.values

    # clip: globally or locally
    mykwargs = kwargs_solve

    if not globalclipping:
        mykwargs["clip_bounds"] = Y_samp.describe().loc[["min", "max"], :].T.values
    else:
        mykwargs["clip_bounds"] = Y.describe().loc[["min", "max"], :].T.values

    # split sample data into train and test
    (
        X_train,
        X_test,
        Y_train,
        Y_test,
        idxs_train,
        idxs_test,
    ) = parse.split_data_train_test(
        X_samp.values, Y_samp.values, frac_test=0.2, return_idxs=True
    )
    latlons_train_samp = this_latlons_samp[idxs_train]

    # solve
    kfold_results_samp = kfold_solve(
        X_train[subset_n, subset_feat],
        Y_train[subset_n],
        solve_function=solve_function,
        num_folds=num_folds,
        return_model=True,
        **mykwargs,
    )

    # return the kfold_results object
    return kfold_results_samp, latlons_train_samp, idxs_train, idxs_test, mykwargs



### One-vs-Rest classifier using Ridge Regression ###


from scipy import linalg
from sklearn.linear_model import Ridge
from sklearn.multiclass import OneVsRestClassifier



def OVR_classifier(
    X_train,
    X_test,
    y_train,
    y_test,
    svd_solve=False,
    lambdas=[1e2],
    return_preds=True,
    return_model=False,
    intercept=False,
    allow_linalg_warning_instances=False,
):
    """Train ridge regression model for a series of regularization parameters.
    Optionally clip the predictions to bounds. Used as the default solve_function
    argument for single_solve() and kfold_solve() below.

    Parameters
    ----------
        X_{train,test} : :class:`numpy.ndarray`
            Features for training/test data (n_obs_{train,test} X n_ftrs 2darray).
        y_{train,test} : :class:`numpy.ndarray`
            Labels for training/test data (n_obs_{train,test} X n_outcomes 2darray).
        svd_solve : bool, optional
            If true, uses SVD to compute w^*, otherwise does matrix inverse for each
            lambda.
        lambdas : list of floats, optional
            Regularization values to sweep over.
        return_preds : bool, optional
            Whether to return predictions for training and test sets.
        return_model : bool, optional
            Whether to return the trained weights that define the ridge regression
            model.
        intercept : bool, optional
            Whether to add an unregulated intercept (or, equivalently, center the X and
            Y data).
        allow_linalg_warning_instances : bool, optional
            If False (default), track for which hyperparameters did ``scipy.linalg`` 
            raise an ill-conditioned matrix error, which could lead to poor performance.
            This is used to discard these models in a cross-validation context. If True,
            allow these models to be included in the hyperparameter grid search. Note
            that these errors will not occur when using ``cupy.linalg`` (i.e. if a GPU 
            is detected), so the default setting may give differing results across 
            platforms.

    Returns
    -------
    dict of :class:`numpy.ndarray`
        The results dictionary will always include the following key/value pairs:
            ``metrics_{test,train}`` : array of dimension n_outcomes X n_lambdas
                Each element is a dictionary of {Out-of,In}-sample model performance 
                metrics for each lambda

        If ``return_preds``, the following arrays will be appended in order:
            ``y_pred_{test,train}`` : array of dimension n_outcomes X n_lambdas
                Each element is itself a 1darray of {Out-of,In}-sample predictions for 
                each lambda. Each 1darray contains n_obs_{test,train} values

        if return_model, the following array will be appended:
            ``models`` : array of dimension n_outcomes X n_lambdas:
                Each element is itself a 1darray of model weights for each lambda. Each 
                1darray contains n_ftrs values
    """
    xp = np
    asnumpy = np.asarray
    # get dimensions needed to shape arrays
    n_ftrs, n_outcomes, n_obs_train, n_obs_test = get_dim_lengths(
        X_train, y_train, y_test
    )
    n_lambdas = len(lambdas)
    # center data if needed
    #X_train, y_train, X_offset, y_offset, _ = _preprocess_data(
    #    X_train, y_train, intercept, normalize=False
    #)
    # set up the data structures for reporting results
    results_dict = _initialize_results_arrays(
        (n_outcomes, n_lambdas), return_preds, return_model
    )
    t1 = time.time()
    # send to GPU if available
    X_train = xp.asarray(X_train)
    y_train = xp.asarray(y_train)

    n_unique_classes = len(np.unique(y_train))

    if n_unique_classes < 2:
        log_text("no variation within model fold", print_text="warn")

    if DEBUG:
        if GPU:
            log_text(f"Time to transfer X_train and y_train to GPU: {time.time() - t1}")
        t1 = time.time()
    # precomputing large matrices to avoid redundant computation - REMOVED 
    # iterate over the lambda regularization values
    training_time = 0
    pred_time = 0
    for lx, lambdan in enumerate(lambdas):   
        # train model
        with warnings.catch_warnings(record=True) as w:
            # bind warnings to the value of w
            warnings.simplefilter("always")
            lambda_warning = False
            clf =  OneVsRestClassifier(Ridge(alpha = lambdan, fit_intercept=intercept)).fit(X_train, y_train)
            # if there is a warning
            if len(w) > 1:
                for this_w in w:
                    log_text(this_w.message)
                # more than one warning is bad
                log_text("warning/exception other than LinAlgWarning", print_text="warn")
            if len(w) > 0:
                # if it is a LinAlgWarning
                if w[0].category == LinAlgWarning:
                    log_text("linalg warning on lambda={0}: ".format(lambdan))
                    # linalg warning
                    if not allow_linalg_warning_instances:
                        log_text("we will discard this model upon model selection")
                        lambda_warning = True
                    else:
                        lambda_warning = None
                        log_text("we will allow this model upon model selection")
                else:
                    [log_text(str(e)) for e in w]
                    log_text("warning/exception other than LinAlgWarning", print_text="warn")
                    
        #####################
        # compute predictions
        #####################
        # send to gpu if available
        X_test = xp.asarray(X_test)
        y_test = xp.asarray(y_test)
        if DEBUG: t5 = time.time()
        # train
        pred_train = clf.predict(X_train) 
        pred_train = y_to_matrix(pred_train)
        # test
        pred_test = clf.predict(X_test)  
        pred_test = y_to_matrix(pred_test)

        # bring back to cpu if needed
        if n_unique_classes > 1:
            model = clf.coef_
            if intercept:
                intercept = clf.intercept_
            else: 
                intercept = 0
        else:
            model = np.full(n_ftrs, None)
            intercept = None

        pred_train, pred_test = asnumpy(pred_train), asnumpy(pred_test)
        y_train, y_test, model = (
            y_to_matrix(asnumpy(y_train)),
            y_to_matrix(asnumpy(y_test)),
            y_to_matrix(asnumpy(model)),
        )
        # create tuple of lambda index to match argument structure
        # of _fill_results_arrays function
        hp_tuple = (lx,)
        # Transpose model results so that n_outcomes is first dimension
        # so that _fill_results_array can handle it
        model = model.T
        # populate results dict with results from this lambda
        results_dict = _fill_results_arrays_clf(
            y_train,
            y_test,
            pred_train,
            pred_test,
            model,
            intercept,
            hp_tuple,
            results_dict,
            hp_warning=lambda_warning,
        )
    if DEBUG:
        log_text(f"Training time: {training_time}")
        log_text(f"Prediction time: {pred_time}" )
        log_text(f"Total time: {time.time() - t1}")
    return results_dict


def compute_metrics_clf(true, pred):
    """ takes in a vector of true values, a vector of predicted values. To add more metrics, 
    just add to the dictionary (possibly with a flag or when
    it is appropriate to add) """
    res = dict()
    res["r2_score"] = sum(true==pred)/true.shape[0]
    return res

def _fill_results_arrays_clf(
    y_train,
    y_test,
    pred_train,
    pred_test,
    model,
    intercept,
    hp_tuple,
    results_dict,
    hp_warning=None,
):
    """Fill a dictionary of results with the results for this particular
    set of hyperparameters.
    Args:
        y_{train,test} (n_obs_{train,test} X n_outcomes 2darray of float)
        pred_{train,test} (n_obs_{train,test} X n_outcomes 2darray of float)
        model (n_outcomes 1darray of arbitrary dtype)
        hp_tuple (tuple): tuple of hyperparameter values used in this model
        results_dict (dict): As created in solve functions, to be filled in.
    """
    n_outcomes = y_train.shape[1]
    for i in range(n_outcomes):
        # get index of arrays that we want to fill
        # first dimension is outcome, rest are hyperparams
        this_ix = (i,) + hp_tuple
        # compute and save metrics
        results_dict["metrics_train"][this_ix] = compute_metrics_clf(
            y_train[:, i], pred_train[:, i]
        )
        results_dict["metrics_test"][this_ix] = compute_metrics_clf(
            y_test[:, i], pred_test[:, i]
        )
        # save predictions if requested
        if "y_pred_test" in results_dict.keys():
            results_dict["y_pred_train"][this_ix] = pred_train[:, i]
            results_dict["y_pred_test"][this_ix] = pred_test[:, i]
        # save model results if requested
        if "models" in results_dict.keys():
            results_dict["models"][this_ix] = model[i]
            results_dict["intercept"][this_ix] = intercept
        # save hp warnings if thats desired
        results_dict["hp_warning"][this_ix] = hp_warning
    return results_dict



def hurdle_model_1(
    X_train,
    X_test,
    y_train_binary, 
    y_test_binary, 
    svd_solve=False,
    thresh = [0.95, 0.90, 0.8, 0.65, 0.5],
    return_preds=True,
    return_model=True,
    allow_convergence_warning_instances=True,
    max_iter = 100,
    C = 1,
    intercept=False,
):
    xp = np
    asnumpy = np.asarray
    # get dimensions needed to shape arrays
    n_ftrs, n_outcomes, n_obs_train, n_obs_test = get_dim_lengths(
        X_train, y_train_binary, y_test_binary
    )
    n_thresh = len(thresh)
    # set up the data structures for reporting results
    results_dict = _initialize_results_arrays(
        (n_outcomes, n_thresh), return_preds, return_model,
    )
    # send to GPU if available
    X_train = xp.asarray(X_train)
    y_train_binary = xp.asarray(y_train_binary)
    # Logit to predict zeros
    
    n_unique_classes = len(np.unique(y_train_binary))

    with warnings.catch_warnings(record=True) as w:

        warnings.simplefilter("always")
        lambda_warning = False
        
        if n_unique_classes > 1:
            zi_1 = LogisticRegression(fit_intercept=intercept,max_iter=max_iter,C=C,
            solver="sag",n_jobs=5).fit(X_train, y_train_binary)
        else:
            log_text("no variation within model fold", print_text="warn")
         # if there is a warning
        if len(w) > 1:
            for this_w in w:
                log_text(this_w.message)
            # more than one warning is bad
            log_text("warning/exception other than ConvergenceWarning", print_text="warn")
        if len(w) > 0:
            # if it is a max iteration warning
            if w[0].category == ConvergenceWarning:
                log_text("ConvergenceWarning warning on C={0}: ".format(C))
                # linalg warning
                if not allow_convergence_warning_instances:
                    raise NotImplementedError("throwing away failed convergence on step 1 hurdle is not implemented or tested")
                    log_text("we will discard this model upon model selection")
                    lambda_warning = True
                else:
                    lambda_warning = None
                    log_text("While there is a ConvergenceWarning, we will still allow this stage 1 hurdle model upon model selection")
            else:
                [log_text(str(e)) for e in w]

                log_text("warning/exception other than ConvergenceWarning", print_text="warn")

        
    # Edit threshold
    if n_unique_classes > 1:
        pred_proba_train = zi_1.predict_proba(X_train)
        pred_proba_test = zi_1.predict_proba(X_test) 
    else:
        # Make sure all the data goes to step 2, which is better at handling this special case
        # where there is no variation in the fold
        pred_proba_train = np.full((n_obs_train,n_outcomes), int(~bool(y_train_binary[0] )))
        pred_proba_test = np.full((n_obs_test,n_outcomes), int(~bool(y_train_binary[0] )))

    for tx, threshold in enumerate(thresh): 
        pred_train = pred_proba_train[:,0] < threshold
        pred_test = pred_proba_test[:,0] < threshold

        # Format output 
        if n_unique_classes > 1:
            model = zi_1.coef_
            intercept = zi_1.intercept_

        else:
            model = np.full(n_ftrs, 0) #no actual model here
            intercept = np.array(None)

        y_train, y_test, pred_train, pred_test, model = (
            y_to_matrix(asnumpy(y_train_binary)),
            y_to_matrix(asnumpy(y_test_binary)),
            y_to_matrix(asnumpy(pred_train)),
            y_to_matrix(asnumpy(pred_test)),
            y_to_matrix(asnumpy(model)),
        )

        # create tuple of C index to match argument structure
        # of _fill_results_arrays function
        hp_tuple = (tx,)

        # Transpose model results so that n_outcomes is first dimension
        # so that _fill_results_array can handle it
        model = model.T
    
        # populate results dict with results from this lambda
        results_dict = _fill_results_arrays_clf(
            y_train,
            y_test,
            pred_train,
            pred_test,
            model,
            intercept,
            hp_tuple,
            results_dict,
            hp_warning=lambda_warning,
        )
    return results_dict

def hurdle_model_2(
    X_train,
    X_test,
    y_train,
    y_test,
    step1_train,
    step1_test,
    zero_rep = 0,
    svd_solve=False,
    lambdas=[100,10,1, 0.1],
    thresh = [0.95, 0.90, 0.8, 0.65, 0.5],
    return_preds=True,
    return_model=True,
    clip_bounds=None,
    allow_linalg_warning_instances=False,
    intercept=False,
    verbose=False,
):
    count = -1 
    xp = np
    asnumpy = np.asarray
    # get dimensions needed to shape arrays
    n_ftrs, n_outcomes, n_obs_train, n_obs_test = get_dim_lengths(
        X_train, y_train, y_test
    )
    n_combos = len(lambdas)*len(thresh)
    # set up the data structures for reporting results
    results_dict = _initialize_results_arrays(
        (n_outcomes, n_combos), return_preds, return_model,
    )
    # send to GPU if available
    X_train = xp.asarray(X_train)
    y_train = xp.asarray(y_train)
    # precomputing large matrices to avoid redundant computation - REMOVED 
    # iterate over the lambda regularization values
    for lx, lambdan in enumerate(lambdas): 
        # Logit to predict zeroes 
        for tx, threshold in enumerate(thresh):
            count = count + 1 
            pred_train_binary = step1_train[0][tx]
            pred_test_binary = step1_test[0][tx]

            train_all_classified_zero = all(pred_train_binary==False)
            test_all_classified_zero = all(pred_test_binary==False)
            
            if train_all_classified_zero:
                log_text("Step 1 hurdle classified all training data as zeros")
            
            if test_all_classified_zero:
                log_text(("Step 1 hurdle classified all test data as zeros \n"
                 "There is no remaining test data for the step 2 ridge model"))

            # For yhat > 0, ridge to predict continuous outcome
            with warnings.catch_warnings(record=True) as w:
                # bind warnings to the value of w
                warnings.simplefilter("always")
                lambda_warning = False
                
                if not train_all_classified_zero: #can't fit model if classifier results leave no data for ridge
                    zi_2 = Ridge(alpha=lambdan, fit_intercept=intercept).fit(
                        X_train[pred_train_binary],
                        y_train[pred_train_binary])
                
                # if there is a warning
                if len(w) > 1:
                    for this_w in w:
                        log_text(this_w.message)
                    # more than one warning is bad
                    raise Exception("warning/exception other than LinAlgWarning")
                if len(w) > 0:
                    # if it is a linalg warning
                    if w[0].category == LinAlgWarning:
                        log_text("linalg warning on lambda={0}: ".format(lambdan))
                        # linalg warning
                        if not allow_linalg_warning_instances:
                            log_text("we will discard this model upon model selection")
                            lambda_warning = True
                        else:
                            lambda_warning = None
                            log_text("we will allow this model upon model selection")
                    else:
                        raise Exception("warning/exception other than LinAlgWarning")
            # Test accuracy
            if verbose:
                log_text("STAGE 2 Lambda " + str(lambdan) + " threshold " + str(threshold) + 
                    " " + str(np.round(zi_2.score(X_test[pred_test_binary], y_test[pred_test_binary]),3)))
            # Make two-step predictions 
            pred_train = pred_train_binary.astype('float')
            zero_idxs_train = (pred_train==0)
            one_idxs_train = (pred_train==1)

            pred_train[zero_idxs_train] = zero_rep

            if not train_all_classified_zero: #can't predict from model if classifier assigned all and model was not fit
                pred_train[one_idxs_train] = np.squeeze(zi_2.predict(X_train[one_idxs_train]))

            pred_test = pred_test_binary.astype('float')
            zero_idxs_test = (pred_test==0)
            one_idxs_test = (pred_test==1)

            pred_test[zero_idxs_test] = zero_rep

            if not test_all_classified_zero: #No need to predict from model if classifier assigned all test vals
                if not train_all_classified_zero: 
                    pred_test[one_idxs_test] = np.squeeze(zi_2.predict(X_test[one_idxs_test]))
                else: #Possible that stage 1 model assigned all train vals but not all test vals
                    pred_test[one_idxs_test] = zero_rep

            if verbose:
                log_text("COMBINED Lambda " + str(lambdan) + " threshold " + str(threshold) + 
                " " + str(np.round(metrics.r2_score(y_test, pred_test),3)))
            # clip if needed
            if clip_bounds is not None:
                for ix, i in enumerate(clip_bounds):
                    # only apply if both bounds aren't None for this outcome
                    if not (i == None).all():
                        pred_train = xp.clip(pred_train, *i) #These are 1d arrays here, in Ridge, they are 2d
                        pred_test = xp.clip(pred_test, *i)
            # Format output
            if not train_all_classified_zero:
                model = zi_2.coef_
                intercept = zi_2.intercept_
            else:
                model = np.full(n_ftrs,None)
                intercept = np.array(None)

            y_train, y_test, pred_train, pred_test, model = (
                y_to_matrix(asnumpy(y_train)),
                y_to_matrix(asnumpy(y_test)),
                y_to_matrix(asnumpy(pred_train)),
                y_to_matrix(asnumpy(pred_test)),
                y_to_matrix(asnumpy(model)),
            )

            # create tuple of C index to match argument structure
            # of _fill_results_arrays function
            hp_tuple = (count,)

            # Transpose model results so that n_outcomes is first dimension
            # so that _fill_results_array can handle it
            
            model = model.T
          

            # populate results dict with results from this lambda
            results_dict = _fill_results_arrays(
                y_train,
                y_test,
                pred_train,
                pred_test,
                model,
                intercept,
                hp_tuple,
                results_dict,
                hp_warning=lambda_warning,
            )
    return results_dict



def new_hurdle_model_2(
    X_train,
    X_test,
    y_train,
    y_test,
    step1_train,
    step1_test,
    zero_rep = 0,
    svd_solve=False,
    lambdas=[100,10,1, 0.1],
    thresh = [0.95, 0.90, 0.8, 0.65, 0.5],
    return_preds=True,
    return_model=True,
    clip_bounds=None,
    allow_linalg_warning_instances=False,
    intercept=False,
    verbose=False,
):
    count = -1 
    xp = np
    asnumpy = np.asarray
    # get dimensions needed to shape arrays
    n_ftrs, n_outcomes, n_obs_train, n_obs_test = get_dim_lengths(
        X_train, y_train, y_test
    )
    n_combos = len(lambdas)*len(thresh)
    # set up the data structures for reporting results
    results_dict = _initialize_results_arrays(
        (n_outcomes, n_combos), return_preds, return_model,
    )
    # send to GPU if available
    X_train = xp.asarray(X_train)
    y_train = xp.asarray(y_train)
    
    y_train_true_positive_idxs = (y_train != zero_rep)
    
    too_little_data_to_fit_Ridge = False
    if sum(y_train_true_positive_idxs) < 2:
        log_text("There are too few true positives to fit a ridge model")
        too_little_data_to_fit_Ridge = True

    # iterate over the lambda regularization values
    for lx, lambdan in enumerate(lambdas): 
        # Logit to predict zeroes 
        for tx, threshold in enumerate(thresh):
            count = count + 1 

            # Classification indices for the train set (from step 1)
            pred_train_binary = step1_train[0][tx]
            
            # Classification indices for the test set (from step 1)
            pred_test_binary = step1_test[0][tx]

            # There are no nonzeros in the test set to evaluate on
            test_all_classified_zero = all(pred_test_binary==False)
            train_all_classified_zero = all(pred_train_binary==False)
            
            if test_all_classified_zero:
                log_text(("Step 1 hurdle classified all test data as zeros \n"
                 "There is no remaining test data for the step 2 ridge model"))

            # For yhat > 0, ridge to predict continuous outcome
            with warnings.catch_warnings(record=True) as w:
                # bind warnings to the value of w
                warnings.simplefilter("always")
                lambda_warning = False
                
                if not too_little_data_to_fit_Ridge: #can't fit model if there is less than 1 true positive left for ridge
                    zi_2 = Ridge(alpha=lambdan, fit_intercept=intercept).fit(
                        X_train[y_train_true_positive_idxs],
                        y_train[y_train_true_positive_idxs])
                
                # if there is a warning
                if len(w) > 1:
                    for this_w in w:
                        log_text(this_w.message)
                    # more than one warning is bad
                    raise Exception("warning/exception other than LinAlgWarning")
                if len(w) > 0:
                    # if it is a linalg warning
                    if w[0].category == LinAlgWarning:
                        log_text("linalg warning on lambda={0}: ".format(lambdan))
                        # linalg warning
                        if not allow_linalg_warning_instances:
                            log_text("we will discard this model upon model selection")
                            lambda_warning = True
                        else:
                            lambda_warning = None
                            log_text("we will allow this model upon model selection")
                    else:
                        raise Exception("warning/exception other than LinAlgWarning")
            # Test accuracy
            if verbose:
                log_text("STAGE 2 Lambda " + str(lambdan) + " threshold " + str(threshold) + 
                    " " + str(np.round(zi_2.score(X_test[pred_test_binary], y_test[pred_test_binary]),3)))
            # Make two-step predictions 
            pred_train = pred_train_binary.astype('float')
            zero_idxs_train = (pred_train==0)
            one_idxs_train = (pred_train==1)

            pred_train[zero_idxs_train] = zero_rep

            if not train_all_classified_zero: #No need to predict from model on train set if classifier assigned all and model was not fit
                pred_train[one_idxs_train] = np.squeeze(zi_2.predict(X_train[one_idxs_train]))

            pred_test = pred_test_binary.astype('float')
            zero_idxs_test = (pred_test==0)
            one_idxs_test = (pred_test==1)

            # If all classified zeros, this line will ensure that those are passed on
            pred_test[zero_idxs_test] = zero_rep

            if not test_all_classified_zero: #No need to predict from model if classifier assigned all test vals
                if not too_little_data_to_fit_Ridge: 
                    pred_test[one_idxs_test] = np.squeeze(zi_2.predict(X_test[one_idxs_test]))
                else: #Possible that stage 1 model assigned all train vals but not all test vals
                    pred_test[one_idxs_test] = zero_rep

            if verbose:
                log_text("COMBINED Lambda " + str(lambdan) + " threshold " + str(threshold) + 
                " " + str(np.round(metrics.r2_score(y_test, pred_test),3)))
            # clip if needed
            if clip_bounds is not None:
                for ix, i in enumerate(clip_bounds):
                    # only apply if both bounds aren't None for this outcome
                    if not (i == None).all():
                        pred_train = xp.clip(pred_train, *i) #These are 1d arrays here, in Ridge, they are 2d
                        pred_test = xp.clip(pred_test, *i)
            # Format output
            if not too_little_data_to_fit_Ridge:
                model = zi_2.coef_
                intercept = zi_2.intercept_
            else:
                model = np.full(n_ftrs,None)
                intercept = np.array(None)

            y_train, y_test, pred_train, pred_test, model = (
                y_to_matrix(asnumpy(y_train)),
                y_to_matrix(asnumpy(y_test)),
                y_to_matrix(asnumpy(pred_train)),
                y_to_matrix(asnumpy(pred_test)),
                y_to_matrix(asnumpy(model)),
            )

            # create tuple of C index to match argument structure
            # of _fill_results_arrays function
            hp_tuple = (count,)

            # Transpose model results so that n_outcomes is first dimension
            # so that _fill_results_array can handle it
            
            model = model.T
          

            # populate results dict with results from this lambda
            results_dict = _fill_results_arrays(
                y_train,
                y_test,
                pred_train,
                pred_test,
                model,
                intercept,
                hp_tuple,
                results_dict,
                hp_warning=lambda_warning,
            )
    return results_dict




def kfold_solve_hurdle(X, y, locations, zero_rep=0, num_folds=5, 
    return_model=False, tunelambda=False, run_fine_tuning_step = False, 
    **kwargs_solve):
    
    """Predict label using a hurdle model. Because of the complexity of the hurdle model, it uses its own solve function. To run a hurdle model, this function should be called instead of `kfold_solve`. 

        Parameters
        ----------
            X : :class:`numpy.ndarray`
                Features for training/test data (n_obs X k_ftrs 2darray).
            y : :class:`numpy.ndarray`
                Labels for training/test data (n_obs X 1).
            locations : :class:`numpy.ndarray`
                    location identifiers for Y values (n_obs X 2 or n_obs x 1; lon, lat coords for tile & polygon_id for polygon)
            return_model : bool, optional
                Whether to return the trained weights that define the ridge regression
                model.
            zero_rep : float/int
                A numeric representation of 0s in the data. When we do log transformations, 0s may be represented as different numbers. In these case, `zero_rep` should be defined using the transformation rule
            tunelambda : bool
                A boolean that indicates whether `kfoldsolve_tunelambda` should be used to automatically tune the ridge hyperparameter. Note that this tuning process will only occur in the second stage. 
            kwarsgs_solve : optional
                Optional set of keyword arguments used for the solve. These are only used in the Ridge (step 2) portion of the hurdle model solve. The exception is the regularization hyperparameter ("C"). If included, this keyword will change the 
regularization added to the logisitic (step 1) regression.
        Returns
        -------
        dict of :class:`numpy.ndarray`
            The results dictionary will always include the following key/value pairs:
                ``metrics_{test,train}`` : array of dimension n_outcomes X n_lambdas
                    Each element is a dictionary of {Out-of,In}-sample model performance 
                    metrics for each lambda

            If ``return_preds``, the following arrays will be appended in order:
                ``y_pred_{test,train}`` : array of dimension n_outcomes X n_lambdas
                    Each element is itself a 1darray of {Out-of,In}-sample predictions for 
                    each lambda. Each 1darray contains n_obs_{test,train} values

            if return_model, the following array will be appended:
                ``models`` : array of dimension n_outcomes X n_lambdas:
                    Each element is itself a 1darray of model weights for each lambda. Each 
                    1darray contains n_ftrs values
    """

    hurdle_type = kwargs_solve.pop("solve_function")

    if hurdle_type not in ["Hurdle","new_Hurdle"]:
        log_text( ("Potentially unclear behavior. kfold_solve_hurdle is called"
                        f" but kwargs_solve['solve_function'] = {hurdle_type}"
                        " Currently, the hurdle model cannot be run using a keyword argument "
                        "so the 'solve_function' key should not be passed into 'kwargs_solve'. "
                        "\n Proceeding with hurdle model solve..."), print_text="warn")
                      
    
    
    if "C" in kwargs_solve.keys(): #make it so that a step1 regularization can be passed in via **kwargs_solve
        C = kwargs_solve.pop("C")
    else:
        C = 1
    
    thresh = kwargs_solve.get("thresh", c.default_hurdle_thresholds)
    
    kwargs_solve["zero_rep"] = zero_rep
    
    Y_binary = (y != zero_rep).astype(int)

    n_unique_classes = len(np.unique(Y_binary))

    if n_unique_classes != 2:
        log_text("can't run hurdle because there are not two classes for Y", print_text="warn")
        log_text("If using continent model, that may not be appropriate")
        log_text("Attempting to proceed...")

    if hurdle_type == "Hurdle":
        step_2_solve_function = hurdle_model_2
    elif hurdle_type == "new_Hurdle":
        step_2_solve_function = new_hurdle_model_2
    
    log_text("Training hurdle model...")
    step1 = kfold_solve(
    X,
    Y_binary,
    locations,
    thresh = thresh,
    solve_function= hurdle_model_1,
    num_folds= num_folds,
    C = C,
    return_preds = True, # we need preds to be fed into step 2; so we hard code as True
    return_model=return_model)

    kwargs_solve["step1"] = step1
    
    if tunelambda:
        step2 = kfoldsolve_tunelambda(
            X,
            y,
            locations,
            solve_function=step_2_solve_function,
            num_folds=num_folds,
            generate_lambda_default=True,
            return_model=return_model,
            run_fine_tuning_step = run_fine_tuning_step,
            # This is now skipped by default
            # run_fine_tuning_step=False, #Step 2 of hurlde is very expensive, we skip lambda fine tuning
            **kwargs_solve,
        )
            
    else:
        step2 = kfold_solve(
        X,
        y,
        locations,
        solve_function=step_2_solve_function,
        num_folds=num_folds,
        return_model=return_model,
        **kwargs_solve,
    )
        
    step2["hurdle_step1_kfold_results"] = step1 #Data from step1 process may be needed for future analysis. Also, saved model weights may be needed.
    
    return step2

## LS: It's very weird that we sometimes use a custom ridge function and sometimes use the sklearn function.
## I assume the MOSAIKS authors have a good reason for this. If we are going to switch to all custom ridge, we
## can use this basic function:
def custom_ridge(X,y, lam, intercept=True):

    X, y, X_offset, y_offset, _ = _preprocess_data(
    X, y, intercept, normalize=False)

    XtX = X.T.dot(X)
    Xty = X.T.dot(y)

    model = linalg.solve(XtX+ lam*np.eye(X.shape[1], dtype=np.float64), 
    Xty)

    intercept_term = y_offset - X_offset.dot(model)

    return model, intercept_term
