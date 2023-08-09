import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg
import pickle
import pandas as pd
import sklearn 
import pandas as pd
from importlib import reload

from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.feature_extraction import DictVectorizer

from scipy.stats import spearmanr

import warnings
import sys

repo_dir = os.environ.get("REPO_DIR")
code_dir = os.path.join(repo_dir, "code/")
data_dir = os.path.join(repo_dir, "data/")

os.chdir(code_dir)

from mosaiks.utils.imports import *



#########################################
#### Solve helpers built on mosaiks pipeline
#########################################


def kfold_solve_custom_split_col(
    X,
    y,
    locations,
    split_col,
    solve_function=solve.ridge_regression,
    num_folds=5,
    return_preds=True,
    return_model=False,
    **kwargs_solve,
):
    """A general skeleton function for computing k-fold cross validation solves.

    Args:
        X (n_obs X n_ftrs 2darray): Feature matrix
        y (n_obs X n_outcomes 2darray): Attribute matrix
        locations (n_obs): location identifiers
        split_col (n_obs X grouping 2darray): Grouping column on which to split X
        solve_function (func): Which solve function in this module will you be using
        num_folds (int): How many folds to use for CV
        return_preds (bool): Return predictions for training and test sets?
        return_model (bool): Return the trained weights that define the ridge regression
            model?
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
    kfold_models = []
    kfold_intercepts = []
    hp_warnings = []
    
    test_numeric_idxs = []
    locations_test = []
    
    print("on fold (of {0}): ".format(num_folds), end="")
    i = 0
    split_col_unique = np.unique(split_col)
    for train_grp_num, val_grp_num in kf.split(split_col_unique):
        print(i, end=" ")
        i += 1
        
        
        train_grp = split_col_unique[train_grp_num]
        val_grp = split_col_unique[val_grp_num]
        
        train_idxs = np.in1d(split_col, train_grp)
        val_idxs = np.in1d(split_col, val_grp)
        
        num_indxs = np.where(val_idxs)[0]
        test_numeric_idxs.append(num_indxs)
        locations_test.append(locations[num_indxs])
        
        

        X_train, X_val = X[train_idxs], X[val_idxs]
        y_train, y_val = y[train_idxs], y[val_idxs]

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
    print("\n")
    rets = {
        "metrics_test": np.array(kfold_metrics_test, dtype=object),
        "metrics_train": np.array(kfold_metrics_train, dtype=object),
        "y_true_test": np.array(kfold_y_test,dtype=object),
        "y_true_train": np.array(kfold_y_train,dtype=object),
        "hp_warning": np.array(hp_warnings,dtype=object),
        "cv": kf,
    }
    
    rets["shuffeled_num_test_indxs"] = np.array(test_numeric_idxs,dtype=object)
    rets["locations_test"] = np.array(locations_test,dtype=object)

    if return_preds:
        rets["y_pred_test"] = np.array(kfold_preds_test,dtype=object)
        rets["y_pred_train"] = np.array(kfold_preds_train,dtype=object)

    if return_model:
        rets["models"] = np.array(kfold_models,dtype=object)
        rets["intercepts"] = np.array(kfold_intercepts,dtype=object)

    return rets



solver_kwargs = {
# set of possible hyperparameters to search over in cross-validation
"lambdas": [1e-3,1e-2, 1e-1, 1e0,1e1,1e2,1e3,1e4,1e5,1e6],
# do you want to return the predictions from the model?
"return_preds": True,
# input the bounds used to clip predictions

# do you want to use an SVD solve or standard linear regression? (NB: SVD is much slower)
"svd_solve": False,
# do you want to allow hyperparameters to be chosen even if they lead to warnings about matrix invertibility?
"allow_linalg_warning_instances": True,

"intercept": True}


def cv_solve(task,
             X_train,
             Y_train,
             solver_kwargs = solver_kwargs, 
             clip_bounds = None,
             plot_scatter= True,
             X_train2 = None,
             lambdas2=None,
            country_fold = False):
    
    assert len(Y_train) == len(X_train) 
    
    if X_train2 is not None:
        assert len(Y_train) == len(X_train2)

    print(task)
    if type(clip_bounds) == dict:
        solver_kwargs["clip_bounds"] = np.array([clip_bounds[task]])
    elif type(clip_bounds) == list and len(clip_bounds) ==2:
        solver_kwargs["clip_bounds"] = np.array([clip_bounds])
    else:
        print("Clip bounds wrong type. No clipping will occur.")

    null_Ys = np.isnan(Y_train)
    if any(null_Ys):
        num_missings = null_Ys.sum()
        print(num_missings)
        warnings.warn(f"{num_missings} missing values for outcome, these will be dropped")

        Y_train = Y_train[~null_Ys]
        X_train = X_train[~null_Ys]

        if X_train2 is not None:
            X_train2 = X_train2[~null_Ys]
    
    locations = np.array(X_train.index)
    print("Training model...")
    print("")

    if country_fold:
        country_arr = pd.Series(X_train.index).apply(lambda x: x[:3]).to_numpy()
    
    if X_train2 is not None:
        num_lams2 = len(lambdas2)


    if X_train2 is None and country_fold:
        
        kfold_results = kfold_solve_custom_split_col(
        np.array(X_train),
        np.array(Y_train),
        locations,
        country_arr,
        solve_function=solve.ridge_regression,
        num_folds=5,
        return_model=True,
        **solver_kwargs
        )
            
            
    elif X_train2 is None and not country_fold:
            kfold_results = solve.kfold_solve(
            np.array(X_train),
            np.array(Y_train),
            locations,
            solve_function=solve.ridge_regression,
            num_folds=5,
            return_model=True,
            **solver_kwargs
            )

    elif country_fold:
        kfold_results_list = []
        for i,ln in enumerate(lambdas2):
            print(f"Rescaling X2 by {ln}")
            print(f"{i} out of {num_lams2}")
            
            this_X_train2= ln * X_train2
            
            this_X = pd.concat([X_train, this_X_train2],axis=1)
            
            kfold_results_temp = kfold_solve_custom_split_col(
            np.array(this_X),
            np.array(Y_train),
            locations,
            country_arr,
            solve_function=solve.ridge_regression,
            num_folds=5,
            return_model=True,
            **solver_kwargs
            )
            
            kfold_results_list.append(kfold_results_temp)
            
        r2_list = [get_r2_from_kfold_results(results) for results in kfold_results_list]
        a_max = np.argmax(r2_list)
        x_rescale_opt = lambdas2[a_max]
        
        print(f"X2 is best rescaled by {x_rescale_opt}")
        kfold_results = kfold_results_list[a_max]
        kfold_results["rescale_X2"]=x_rescale_opt
        if a_max == 0:
            warnings.warn("Rescale X2 val is at min of range")
        if a_max == (num_lams2-1):
            warnings.warn("Rescale X2 val is at max of range")

    else:
        kfold_results_list = []
        for i,ln in enumerate(lambdas2):
            print(f"Rescaling X2 by {ln}")
            print(f"{i} out of {num_lams2}")
            
            this_X_train2= ln * X_train2
            
            this_X = pd.concat([X_train, this_X_train2],axis=1)
            
            kfold_results_temp = solve.kfold_solve(
            np.array(this_X),
            np.array(Y_train),
            locations,
            solve_function=solve.ridge_regression,
            num_folds=5,
            return_model=True,
            **solver_kwargs
            )
            
            kfold_results_list.append(kfold_results_temp)
        
        r2_list = [get_r2_from_kfold_results(results) for results in kfold_results_list]
        a_max = np.argmax(r2_list)
        x_rescale_opt = lambdas2[a_max]
        
        print(f"X2 is best rescaled by {x_rescale_opt}")
        kfold_results = kfold_results_list[a_max]
        kfold_results["rescale_X2"]=x_rescale_opt
        if a_max == 0:
            warnings.warn("Rescale X2 val is at min of range")
        if a_max == (num_lams2-1):
            warnings.warn("Rescale X2 val is at max of range")

    
    best_lambda_idx, best_metrics, best_preds = ir.interpret_kfold_results(
    kfold_results, "r2_score", hps=[("lambdas", solver_kwargs["lambdas"])], 
    save_weight_path = None
    )

    # Now we are going to fit a model with the opt hyperparameter and ALL the data
    # This is the model we will use for running preds
    best_lambda = solver_kwargs["lambdas"][best_lambda_idx[0][0]]

    if X_train2 is not None:
        kfold_results["prediction_model_opt_rescale_of_X2"] = x_rescale_opt
        X_train2 = X_train2 * x_rescale_opt
        X_train = pd.concat([X_train,X_train2],axis=1)


    #To do, change this so we use custom ridge function OR sklearn, weird to do both
    mod = Ridge(fit_intercept=solver_kwargs["intercept"], alpha = best_lambda )
    mod.fit(X_train, Y_train)

    kfold_results["prediction_model_weights"] = mod.coef_
    kfold_results["prediction_model_intercept"] = mod.intercept_
    kfold_results["clip_bounds"] = clip_bounds


    
    #avg_r2_score = np.mean([metric[0]["r2_score"] for metric in best_metrics])
    
    if plot_scatter:
        plots.performance_density(
        kfold_results,
        "Ridge",
        val=task,
        save_dir=None,
        app_name=task,
        kind="scatter",
        suffix = None,
        )
    
    return kfold_results



#########################################
#### Functions to interpret solve data outputs (kfold_results objects)
#########################################

def get_r2_from_kfold_results(kfold_results):

    truth, preds = get_truth_preds_from_kfold_results(kfold_results)

    return sklearn.metrics.r2_score(truth, preds)


def predict_y_from_kfold_dict(X,kfold_dict,task, X2=None, clip_preds=True):
    """
    This function calculates a predicted value from a kfold_dictionary

    It requires that a final model was run with all of the outcomes included.


    """
    kfold_results = kfold_dict[task]
    # indxs = np.hstack(kfold_results['locations_test'])
    
    if X2 is not None:
        X2 = X2 * kfold_results["prediction_model_opt_rescale_of_X2"]
        
        X = pd.concat([X,X2], axis=1)
    
    # X = X.loc[indxs].sort_index()
    
    preds = X.dot(kfold_results["prediction_model_weights"]) + kfold_results["prediction_model_intercept"]
    
    ## Now do clipping
    if clip_preds:
        preds = preds.clip(*kfold_results["clip_bounds"])

    return preds


def get_truth_preds_from_kfold_results(kfold_results):
    
    _, _, best_preds = ir.interpret_kfold_results(
    kfold_results, crits="r2_score"
    )
    
    truth = np.vstack(
        [solve.y_to_matrix(i) for i in kfold_results["y_true_test"].squeeze()]
    )
    
    preds = np.vstack([solve.y_to_matrix(i) for i in best_preds.squeeze()])
    
    return truth, preds



def get_within_perf_from_kfold_dict(kfold_dict, 
    task, metric="ALL", scatter=False, 
    method="calc", return_df=False, truth=None, preds=None, 
                                    demeaned_input=False, not_demeaned_df=None,
                                   eval_select_countries=None, logged=False):
    """
    Metric must be "pearson", "r2_score", or "spearman", or "ALL". If the latter, we get a dictionary of 6 metrics.
    
    Optional method to run scatter for demeaned preds
    
    Optional method to just return the whole df
    pyth
    Only works with calculated means, not population weighted means
    
    """
    
    if method != "calc":
        raise NotImplementedError
    
    if truth is None and preds is None:

        kfold_results = kfold_dict[task]  
        truth, preds = get_truth_preds_from_kfold_results(kfold_results)
        locations = np.hstack(kfold_results["locations_test"])
        
    else:
        locations = truth.index
        assert all(locations == preds.index)
        truth, preds = truth.to_numpy(), preds.to_numpy()
        
    
    df = pd.DataFrame({"preds":preds.flatten(), "truth": truth.flatten()}, index = locations)
    
    # use the strings in the indices to ID countries
    df["country"] = pd.Series(df.index).apply(lambda x: x[:3]).to_numpy()
    
    avg = df["truth"].mean()
    
    if eval_select_countries is not None:
        df = df[df["country"].isin(eval_select_countries)]
    
    ### As a first step, we want to add back the country means. This is the mean of the ADM1 observations
    ## aggregated to the ADM1 level.
    if demeaned_input and not_demeaned_df is not None:
        adm0_means = not_demeaned_df.groupby("ISO_Code")[task].mean().rename("add_back_country_mean")
        df = df.merge(adm0_means, "left", left_on="country", right_index=True)
        if logged:
            df["add_back_country_mean"] = np.log(df["add_back_country_mean"])
        df["preds"] = df["preds"] + df["add_back_country_mean"]
        df["truth"] = df["truth"] + df["add_back_country_mean"]
        
            
    
    # Note that this is the mean of the observations, different from the true pop weighted mean
    ## This demean step is redundant for the demeaned_input models. Still, it's here for clarity
    country_mean_short = df.groupby("country")[["truth","preds"]].mean().rename(columns = {
        "truth": "true country mean", "preds":"pred country mean"})
    
    df = df.merge(country_mean_short, how="left", left_on="country", right_index=True)
    
    df["preds_demean"] = df["preds"] - df["pred country mean"]
    df["true_demean"] = df["truth"] - df["true country mean"]
    
    if scatter:
        make_train_pred_scatterplot(task + " - within ADM0 scatter",df["true_demean"],df["preds_demean"])
    
    if return_df:
        return df
    
    within_r2 = sklearn.metrics.r2_score(df["true_demean"],df["preds_demean"])
    if metric == "r2_score":
        return within_r2
    
    within_pearson = np.corrcoef(df["true_demean"],df["preds_demean"])[0,1] ** 2
    if metric == "pearson":
        return within_pearson
    
    within_spearman = spearmanr(df["true_demean"],df["preds_demean"]).correlation
    if metric == "spearman":
        return within_spearman
    
    if metric == "ALL":
        r2 = sklearn.metrics.r2_score(df["truth"],df["preds"])
        pearson = np.corrcoef(df["truth"],df["preds"])[0,1] ** 2
        spearman = spearmanr(df["truth"],df["preds"]).correlation
    
        output_dict = {"pearson" : pearson, "spearman" : spearman, "r2" : r2,
                       "within_adm0_pearson": within_pearson, "within_adm0_spearman": within_spearman, "within_adm0_r2":within_r2}
        
        return output_dict
    


#########################################
#### Other general purpose utils
#########################################



def test_train_split(df,test_frac=.2,seed=333):
    """
    Takes a pd.DataFrame and returns training and test indices.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    y_hat : array or list (1 dimension)
        Predicted values
    test_frac : float
        fraction of data to be left in the test set. Must be between 0 and 1.
    seed : int
        Random seed to be used for replicability
        
    Returns
    -------
    train_indxs, test_indxs : tuple of pd.Index objects
    """
    
    rs = np.random.RandomState(seed)

    n = len(df)
    shuffled_idxs = rs.choice(n,n, replace=False)

    test_idxs = shuffled_idxs[:int(n*test_frac)]
    train_idxs = shuffled_idxs[int(n*test_frac):]
    
    return df.index[train_idxs], df.index[test_idxs]


def get_test_idxs():
    test_idxs = np.load(data_dir + "int/Train_Test_split/test_idxs.npy", allow_pickle=True)
    return test_idxs




#########################################
### Plotting utils
#########################################

def make_train_pred_scatterplot(task, y_test, preds_test, x_label = "RCF preds", additional_title_text = "", verbose=True, alpha =.2):
    """
    Create standard scatterplot figure comparing test predictions to truths.
    
    Parameters
    ----------
    task : str
        String to be appended to the scatter title
    y_test : array or list (1 dimension)
        Truth values
    preds_test : array or list (1 dimension)
        Predicted values
    x_label : str
        String such that the prediction axis can have a new label
    additional_title_text : str
        Text that can optionally be appended to the title of the scatterplot 
    verbose: bool
        If True R2 and Pearson R are printed after being calculated.
        
    Returns
    -------
    None
    
    """
    
    test_r2 = sklearn.metrics.r2_score(y_test,preds_test)
    test_pearson_r = np.corrcoef(y_test, preds_test)[0,1]
    test_spearman_r = scipy.stats.spearmanr(y_test, preds_test).correlation
    
    if verbose:
        print('holdout r2: {0:.3f}'.format(test_r2))
        print('holdout Pearson R: {0:.3f}'.format(test_pearson_r))
        print("\n")

    fig, ax = plt.subplots( figsize=(10,10))
    ax.scatter(preds_test, y_test, alpha=alpha)
    ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
    ax.set_xlabel(x_label)
    ax.set_ylabel('labels (truth)')
    title_str = task + "\n" + "$R^2$ = {0:.3f} \n Pearson R = {1:.3f}".format(test_r2,test_pearson_r)
    title_str = title_str + "\n" + "Spearman R = {0:.3f}".format(test_spearman_r)
    
    ax.set_title(additional_title_text+title_str)
    ax.plot([y_test.min(), y_test.max()],[y_test.min(), y_test.max()], color='black')  

        
#########################################
#### Demeaning tools
#########################################

def df_to_demeaned_y_vars(task, 
                          df, 
                          method ="calc", 
                log_before_diff=False):
    
    hats = df.copy()
    hats.rename(columns = {task : "y_true"}, inplace=True)

    if method == "calc":
        country_means = hats.groupby("ISO_Code")["y_true"].mean().to_frame().rename(columns={"y_true" : "y_bar_country"})
        hats = hats.merge(country_means, "left", left_on = "ISO_Code", right_index=True)
    else:
        raise Exception("NotImplemented  - Invalid method input")
    
    hats["demeaned_y_true"] = hats.y_true - hats.y_bar_country
    
    if log_before_diff:
        hats["demeaned_y_true"] = np.log(hats.y_true) - np.log(hats.y_bar_country)
    return hats["demeaned_y_true"]


def X_matrix_to_demeaned_X(X, method="calc", return_mean_frame=False):  
    X_copy = X.copy()
    countries = pd.Series(X.index).apply(lambda x : x[:3] ).to_frame()\
    .set_index(X.index).rename(columns = {X.index.name: "country"})
    
    X_copy["country"]= countries
    
    if method == "calc":
        country_feat_means = X_copy.groupby("country").mean()
        if return_mean_frame:
            return country_feat_means
    
    else:
        raise Exception("NotImplemented")
    
    mean_matrix = X_copy[["country"]].merge(country_feat_means, "left", left_on="country",right_index=True).drop("country",axis=1)
    
    assert mean_matrix.shape == X.shape
    
    return X - mean_matrix



def generalized_demean(df1, df2, df1_colname_for_demeaning):
    """
    function to demean df1 by df2. 
    
    Requires a colname for df1 that matches the index for df2
    """
    df1 = df1.copy()
    
    long_col = df1.pop(df1_colname_for_demeaning)
    
    df_2_long_frame = long_col.to_frame().merge(df2,"left", left_on=df1_colname_for_demeaning,right_index=True)
    long_mean_x_arr = df_2_long_frame.drop(columns = df1_colname_for_demeaning).to_numpy()
    
    assert df1.shape == long_mean_x_arr.shape
    
    demean_X_arr = df1.to_numpy() - long_mean_x_arr
    
    return pd.DataFrame(demean_X_arr, index = df1.index, columns=df1.columns)



### Raster utils

def rasterize_df(df,data_colname, grid_delta = .01, lon_col="lon", lat_col = "lat", custom_extent=None):
    half_grid_delta = grid_delta/2
    
    if custom_extent is None:
        min_lon, max_lon = df[lon_col].min() - half_grid_delta, df[lon_col].max() + half_grid_delta
        min_lat, max_lat = df[lat_col].min() - half_grid_delta, df[lat_col].max() + half_grid_delta
    else:
        min_lon, min_lat = np.array(custom_extent)[[0,2]] 
        max_lon, max_lat = np.array(custom_extent)[[1,3]]

    y_shape = int(np.floor((max_lat - min_lat)/grid_delta))
    x_shape = int(np.floor((max_lon - min_lon)/grid_delta))

    raster = np.full( (y_shape, x_shape), np.nan)

    label_areas =  (np.round( (df[lat_col]- min_lat-half_grid_delta)/grid_delta).astype(int), 
                    np.round( (df[lon_col]- min_lon-half_grid_delta)/grid_delta).astype(int)
                   )

    raster[label_areas] = df[data_colname]

    extent = np.array([min_lon, 
                       max_lon,
                       min_lat,
                       max_lat])

    raster = np.flip(raster,axis=0)
    
    return raster, extent


def flatten_raster(arr, transform):
    """
    Currently only buily to work with rasters that are in degree units
    
    """
    
    num_rows = arr.shape[0]    
    num_cols = arr.shape[1]
    row_range = (0,np.arange(0, num_rows))
    col_range = (np.arange(0, num_cols),0)

    Xs = (transform * transform.translation(0.5,0.5) * col_range)[0]
    Ys = (transform * transform.translation(0.5,0.5) * row_range)[1]
    
    x,y = np.meshgrid(np.array(Xs),np.array(Ys))
    x,y = x.flatten(), y.flatten()
    
    vals = arr.flatten()
    
    return x, y, vals



##### Raster downsclaing function

from numpy.lib.stride_tricks import as_strided as ast

def norm_shape(shape):
    '''
    Normalize numpy array shapes so they're always expressed as a tuple, 
    even for one-dimensional shapes.
     
    Parameters
        shape - an int, or a tuple of ints
     
    Returns
        a shape tuple
    '''
    try:
        i = int(shape)
        return (i,)
    except TypeError:
        # shape was not a number
        pass
 
    try:
        t = tuple(shape)
        return t
    except TypeError:
        # shape was not iterable
        pass
     
    raise TypeError('shape must be an int, or a tuple of ints')
 

def sliding_window(a,ws,ss = None,flatten = True):
    '''
    Return a sliding window over a in any number of dimensions
     
    Parameters:
        a  - an n-dimensional numpy array
        ws - an int (a is 1D) or tuple (a is 2D or greater) representing the size 
             of each dimension of the window
        ss - an int (a is 1D) or tuple (a is 2D or greater) representing the 
             amount to slide the window in each dimension. If not specified, it
             defaults to ws.
        flatten - if True, all slices are flattened, otherwise, there is an 
                  extra dimension for each dimension of the input.
     
    Returns
        an array containing each n-dimensional window from a
    '''
     
    if None is ss:
        # ss was not provided. the windows will not overlap in any direction.
        ss = ws
    ws = norm_shape(ws)
    ss = norm_shape(ss)
     
    # convert ws, ss, and a.shape to numpy arrays so that we can do math in every 
    # dimension at once.
    ws = np.array(ws)
    ss = np.array(ss)
    shape = np.array(a.shape)
     
     
    # ensure that ws, ss, and a.shape all have the same number of dimensions
    ls = [len(shape),len(ws),len(ss)]
    if 1 != len(set(ls)):
        raise ValueError
     
    # ensure that ws is smaller than a in every dimension
    if np.any(ws > shape):
        raise ValueError
    # how many slices will there be in each dimension?
    newshape = norm_shape(((shape - ws) // ss) + 1)
    # the shape of the strided array will be the number of slices in each dimension
    # plus the shape of the window (tuple addition)
    newshape += norm_shape(ws)
    # the strides tuple will be the array's strides multiplied by step size, plus
    # the array's strides (tuple addition)
    newstrides = norm_shape(np.array(a.strides) * ss) + a.strides
    strided = ast(a,shape = newshape,strides = newstrides)
    if not flatten:
        return strided
     
    # Collapse strided so that it has one more dimension than the window.  I.e.,
    # the new array is a flat list of slices.
    meat = len(ws) if ws.shape else 0
    firstdim = (np.product(newshape[:-meat]),) if ws.shape else ()
    dim = firstdim + (newshape[-meat:])
    
    # remove any dimensions with size 1
    #dim = filter(lambda i : i != 1,dim)
    
    return strided.reshape(dim)


def resize_2d_nonan(array,factor,func= np.median):
    """
    Resize a 2D array by different factor on two axis skipping NaN values.
    If a new pixel contains only NaN, it will be set to NaN
    Parameters
    ----------
    array : 2D np array
    factor : int or tuple. If int x and y factor wil be the same
    Returns
    -------
    array : 2D np array scaled by factor
    Created on Mon Jan 27 15:21:25 2014
    @author: damo_ma
    """
    xsize, ysize = array.shape

    if isinstance(factor,int):
        factor_x = factor
        factor_y = factor
        window_size = factor, factor
    elif isinstance(factor,tuple):
        factor_x , factor_y = factor
        window_size = factor
    else:
        raise NameError('Factor must be a tuple (x,y) or an integer')

    if (xsize % factor_x or ysize % factor_y) :
        raise NameError('Factors must be integer multiple of array shape')

    new_shape = int(xsize / factor_x), int(ysize / factor_y)
    
    print(new_shape)
    # non-overlapping windows of the original array
    windows = sliding_window(array, window_size)
    # windowed boolean array for indexing
    notNan = sliding_window(np.logical_not(np.isnan(array)), window_size)

    #list of the means of the windows, disregarding the Nan's
    means = [func(window[index]) for window, index in zip(windows, notNan)]
    # new array
    new_array = np.array(means).reshape(new_shape)

    return new_array


def upscale_grid_vector(x, decimal_place=1):
    
    """
    Takes a vector of grid centroids and reformats them to be grid centroids for a more coarse grid.
    
    Decimal place specifies how coarse the grid should be. 
    
    decimal_place=2 : .01 x. 01 degree grid
    decimal_place=1 : .1 x. 1 degree grid
    decimal_place=0 : 1 x. 1 degree grid
    decimal_place=-1 :  10 x. 10 degree grid
    
    """
    dec = np.round(.1**decimal_place, decimal_place)
    half_dec = dec/2 
    
    return np.round(np.round(x - half_dec,decimal_place)+half_dec, decimal_place+1)


