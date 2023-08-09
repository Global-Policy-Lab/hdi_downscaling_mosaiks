

import numpy as np
import time

from mosaiks.solve import solve_functions as solve
from mosaiks.utils.logging import log_text as log_text
from mosaiks import transforms

import copy

solver_interpret = {"Ridge": solve.ridge_regression,
                    "OVR_classifier" : solve.OVR_classifier,
                    "Hurdle": "Hurdle", #Hurdle implementation is more complicated.
                    "new_Hurdle": "new_Hurdle"} #new Hurdle trains second stage on true positives

def set_bounds(c_app, Y_train):
    """
    Returns a two item list. It needs to further transformed in order to be passed into the solve function.
    """
    bounds = c_app.get("bounds_pred", "auto")  #default behavior is to auto set clip bounds
    
    replace_none = c_app.get("bound_replace_None_w_observed", True)# default to True

    if isinstance(bounds, list) and len(bounds)==2:
        if replace_none:
            if bounds[0] is None:
                bounds[0] = np.min(Y_train)
            if bounds[1] is None:
                bounds[1] = np.max(Y_train)

    elif bounds == "auto" or ( (bounds is None) and (replace_none) ):
        bounds = [np.min(Y_train), np.max(Y_train)]
    
    elif bounds is None:
        bounds = [None, None]
        
    else:
        raise Exception( 
            ("`bounds_pred` input not understood." 
             "Check that you have specified a two item list, the string `auto`, or None"))

    return bounds

def interpret_solve_options(c, c_app):
    """
    Helper function to interpret the different solver options that are set in
    c_app.

    Parameters:
    -----------
    return_solver_kwargs: boolean. Return the default solver_kwargs dictionary?
        This is here because during autotuning, we set the solver_kwargs
        dictionary early on and don't want to modify it again.
    """
    tunelambda = c_app.get("tunelambda", "coarse")

    if tunelambda not in ["coarse", "fine", False, None]:
        log_text("tunelambda argument not understood and is likely deprecated")
        log_text("Defaulting to coarse lambda tuning")
        tunelambda = "coarse"

    fine_tune_lambda = True if tunelambda == "fine" else False
    
    model = c_app.get("solve_function", "Ridge")
    
    polygon_id = c_app.get("polygon_id_colname")
    
    labels_file = c_app["labels_filename"]
    

    #Use intercept as default behavior
    intercept = c_app.get("intercept", True)

    # interpret solver and modify solver_kwargs accordingly
    solver = solver_interpret[model]
    
    hurdle = solver in ["Hurdle", "new_Hurdle"] #boolean for whether we are running a hurdle
    
    region_type = c_app.get("region_type")
    
    solver_kwargs = { 
        # These are harcoded, but they *could* be added as optional params in c_app
        "return_preds": True, # do you want to return the predictions from the model?
        "return_model": True, #Do you want to return model weights and intercept?
        "svd_solve": False, # do you want to use an SVD solve? Not implemented for all models
        "allow_linalg_warning_instances": False, # do you want to allow hyperparameters to be chosen even if they lead to matrix inversion warnings?
        "intercept" : intercept, # Use an intercept when fitting the model?
    }

    if hurdle:
        thresh = c_app.get("hurdle_threshholds", c.default_hurdle_thresholds)
        solver_kwargs["thresh"] = thresh
        solver_kwargs["solve_function"] = solver

    num_folds = 5
    

    return (tunelambda, fine_tune_lambda, model, polygon_id, labels_file, intercept, solver, 
                hurdle, region_type, num_folds, solver_kwargs)





def master_kfold_solve(c, c_app, X, y, locations, regions=None,
                       DEBUG_CONTINENT=False):
    """
    This is a wrapper to run the various different solve functions. 
    
    The goal is for all model specifications to be passed in via c_app. This function can then run predictions for all types
    of models. When doing autotuning, model specifications can be changed by adusting the c_app specifications.
    
    Because log transformation choice is now a model tuning parameter, this function applies the transformation based on the 
    c_app `transformation` key.
    
     Specifically, this function gives the option run the lambda autotuning solve function, the hurdle solve function, and the 
     continent based solve function.
       
       
     Parameters
    ----------
        c (config module): The module you get when importing config.py
        c_app: dict : Full dicionary of configuration parameters for the label and model
        X : :class:`numpy.ndarray`
            Features for training/test data (n_obs_{train,test} X n_ftrs 2darray).
        y : :class:`numpy.ndarray`
            Labels for training/test data (n_obs_{train,test} X n_outcomes 2darray).
        locations : :class:`numpy.ndarray`
            Locations for training data (tile: n_obs X n_outcomes 2darray; polygon: n_obs X n_outcomes adarray)
        regions: :class:`numpy.ndarray`
            Regions for training data
       
     Retruns
    ----------
    dict of :class:`numpy.ndarray` (or list of dictionaries for continent model)
        The results dictionary will always include the following key/value pairs:
            ``metrics_{test,train}`` : array of dimension n_outcomes X n_lambdas
                Each element is a dictionary of {Out-of,In}-sample model performance
                metrics for each lambda

    """
    start_time = time.time()

    #fixes strange behavior where c_app is modifed when at.tunelambda() calls this function
    c_app = copy.deepcopy(c_app)

    if not (len(y) == len(X) == len(locations)):
        log_text("Cannot run solve. Input variables have misaligned lengths")
        log_text(f"len(X) = {len(X)}, len(y)={len(y)}, len(locations)={len(locations)}")
        raise Exception

    (tunelambda, fine_tune_lambda, model, _, _, _, solver, hurdle,
     region_type, num_folds, solver_kwargs
    ) = interpret_solve_options(c, c_app)        

    bounds = set_bounds(c_app,y) # This function automatically looks for whether c_app specifies replace None
  
    zero_rep = 0

    if model != "OVR_classifier":
        transformation = c_app.get("transformation")

        #As currently implemented, transformation MUST happen here. Otherwise we would log twice...
        # To prevent future errors, we have removed the transform step from elsewhere in the pipeline

        if transformation == "log":
            
            y_has_zeros = False
            if y.min() <= 0:
                y_has_zeros = True

            y, pre_transform_added_value = transforms.custom_log_transform(y)
            log_text("bounds without transform: " + str(bounds))

            if y_has_zeros:
                zero_rep = np.log(pre_transform_added_value + 0)

            bounds[0] = np.log(pre_transform_added_value + bounds[0])
            bounds[1] = np.log(pre_transform_added_value + bounds[1])

            log_text("bounds with transform: " + str(bounds))

            c_app["pre_transform_added_value"] = pre_transform_added_value

        elif transformation == "IHS":
            y = np.arcsinh(y)
            bounds = np.arcsinh(bounds)


        # finish setting bounds
        log_text(f"clip bounds set to {bounds}")
        solver_kwargs["clip_bounds"] = np.array([bounds])
    
    if region_type:
        if regions is None:
            log_text("Cannot run continent model without regions. Raising exception", print_text="warn")
            raise Exception
        
        if len(regions) != len(y):
            log_text("regions var has misaligned length", print_text="warn")
            log_text(f"len(regions) = {len(regions)}, len(y) = {len(y)}")
            raise Exception

        results_list = run_region_models(c, c_app, 
                                               X,
                                               y,
                                               locations,
                                               regions,
                                               zero_rep,
                                               DEBUG=DEBUG_CONTINENT,
                                               **solver_kwargs)
        return results_list
        
    elif hurdle:
        kfold_results = solve.kfold_solve_hurdle(
            X,
            y,
            locations,
            zero_rep=zero_rep, 
            num_folds=num_folds, 
            tunelambda=tunelambda,
            run_fine_tuning_step = fine_tune_lambda,
            **solver_kwargs)
        
    elif tunelambda:
        kfold_results = solve.kfoldsolve_tunelambda(X, 
                                    y,
                                    locations,
                                    solve_function=solver,
                                    num_folds=num_folds,
                                    generate_lambda_default=True,
                                    run_fine_tuning_step = fine_tune_lambda,
                                    fit_model_after_tuning = True,
                                    **solver_kwargs)
    else:
        if "lambdas" not in solver_kwargs:
            log_text("Autotune set to False and no lambdas given. Setting lambda range from 0.001 to 1000")
            solver_kwargs["lambdas"] = np.logspace(-3, 3, 9) #default lam search for non tuning
            
        kfold_results = solve.kfold_solve(X,
                                  y,
                                  locations,
                                  solve_function=solver,
                                  num_folds=num_folds,
                                  fit_model_after_tuning = True,
                                  **solver_kwargs)

    end_time = time.time()
    solve_time_elapsed = round((end_time - start_time)/60)
    log_text("Solve wall minutes elapsed = {}".format(solve_time_elapsed))
    kfold_results["solve_wall_time_minutes"] = solve_time_elapsed
    # Let's add solve details to the data output
    kfold_results["c_app"] = c_app 
    kfold_results["solver_kwargs"] = solver_kwargs

    return kfold_results



def run_region_models(c, c_app, X, y, locations, regions, zero_rep=0,
                      DEBUG=False, **solver_kwargs):
    """
    Function to run regional models. This function works for
    continent and country models. It could easily be extended to other regions.

    This function takes in fully prepped (i.e. merged, split, and transformed)
    data, subsets the data to continents, and returns kfold_results objects
    with results for each continent

    Parameters
    ----------
        c (config module) : The module you get when importing config.py
        c_app: dict : Full dicionary of configuration parameters for the
            label and model
        X : :class:`numpy.ndarray`
            Features for training/test data (n_obs_{train,test} X n_ftrs 2darray).
        y : :class:`numpy.ndarray`
            Labels for training/test data (n_obs_{train,test} X n_outcomes 2darray).
        locations : :class:`numpy.ndarray`
            Locations for training/test data (tile: n_obs X n_outcomes
                2darray; polygon: n_obs X n_outcomes adarray)
        regions : :class:`numpy.ndarray`
            Regions for training/test data
        zero_rep : float/int
                A numeric representation of 0s in the data. When we do log 
                transformations, 0s may be represented as different numbers.
                In these case, `zero_rep` should be defined using the
                transformation rule.
        DEBUG: bool
            Subset the data for faster debugging?
        solver_kwargs (dict): Parameters to pass to the solve func
    Returns
    -------
        A list of kfold_results objects, where each item in the list corresponds
        to the results for that region.

    """

    results_list = []
    
    regions_unique = np.unique(regions)

    # for r in r_dict.keys():
    for r in regions_unique:
        start_time = time.time()
        log_text("Running {}".format(r))
        mask = (regions == r)

        X_r = X[mask,:]
        y_r = y[mask]
        locations_r = locations[mask]
        
        (tunelambda, fine_tune_lambda, _, _, _, _, solver, _,
         _, num_folds, _
        ) = interpret_solve_options(c, c_app)

        # skip if there's not enough data for the region
        # For classifier (string) types
        if y_r.dtype == 'O':
            if y_r.size < num_folds * 2:
                log_text(
                f'Only {len(y_r)} obs, not enough to run {num_folds}-fold CV.' 
                + 'Skipping this region',
                print_text='warn'
            )
                log_text(f"Dropping {len(y_r)} observations from {r}",
                         print_text="warn")

                continue
        # for everything else (numeric)
        elif all(np.isnan(y_r)):
                log_text('All data for {} is NaN. Skipping.'.format(r))
                continue
        elif len(np.unique(y_r)) == 1:
                log_text('No variation in {}. Skipping.'.format(r), print_text="warn")
                log_text("Use of continent model may not be appropriate...",print_text="warn")
                log_text(f"Dropping {len(y_r)} observations from {r}",
                         print_text="warn")
                continue
        elif len(y_r) < num_folds * 2:
            log_text(
                f'Only {len(y_r)} obs, not enough to run {num_folds}-fold CV.' 
                + 'Skipping this region',
                print_text='warn'
            )
            
            log_text(f"Dropping {len(y_r)} observations from {r}",
                         print_text="warn")
            continue

        if DEBUG:
            X_r = X_r[:1000] 
            y_r = y_r[:1000]
            locations_r = locations_r[:1000]
            regions_r = regions_r[:1000]

        if c_app.get('solve_function') in ["Hurdle", "new_Hurdle"]:
            these_results = solve.kfold_solve_hurdle(X_r,
                                                     y_r,
                                                     locations_r,
                                                     zero_rep = zero_rep,
                                                     num_folds=num_folds,
                                                     tunelambda=tunelambda,
                                                     run_fine_tuning_step = fine_tune_lambda,
                                                     **solver_kwargs)  
        
        else:
            if tunelambda:
                these_results = solve.kfoldsolve_tunelambda(X_r,
                                                            y_r,
                                                            locations_r,
                                                            solve_function=solver,
                                                            num_folds=num_folds,
                                                            run_fine_tuning_step = fine_tune_lambda,
                                                            fit_model_after_tuning = True,
                                                            **solver_kwargs)
            else:
                if "lambdas" not in solver_kwargs:
                    log_text("Autotune set to False and no lambdas given. Setting lambda range from 0.001 to 1000")
                    solver_kwargs["lambdas"] = np.logspace(-3, 3, 9) #default lam search for non tuning
                these_results = solve.kfold_solve(X_r,
                                                  y_r,
                                                  locations_r,
                                                  solve_function=solver,
                                                  num_folds=num_folds,
                                                  fit_model_after_tuning = True,
                                                  **solver_kwargs)

        these_results["region"]  = r

        end_time = time.time()
        region_time_elapsed = ((end_time - start_time)/60)
        log_text("region solve wall minutes elapsed = {:.0f}".format(region_time_elapsed))
        these_results["region_solve_wall_time_minutes"] = region_time_elapsed

        # Let's add solve details to the data output
        these_results["c_app"] = c_app
        these_results["solver_kwargs"] = solver_kwargs



        results_list += [these_results]
        # append results to existing results dictionary
        log_text("Done with {}".format(r))
    
    return results_list

