
from os.path import join, exists
from os import listdir, mkdir

import numpy as np
import pandas as pd
import pickle
import zarr

from sklearn.metrics import r2_score

from mosaiks import config as c
from mosaiks.solve import interpret_results as ir
from mosaiks.solve import data_parser as parse
from mosaiks.utils import io
from mosaiks import transforms

from mosaiks.utils.config_read import extract_config
import mosaiks.utils.io as mio

from mosaiks import transforms

import time

from mosaiks.utils.logging import log_text

APP_DATA_PATH = "/shares/maps100/data/output/Xb_betas/Xb_betas.p"

XB_PRED_DIRECTORY = "/shares/maps100/data/output/applications"

def get_all_grid_paths(grid):
    if grid == "global_sparse":    
        local_paths = [join(c.features_dir, "global_sparse_grid", f"{grid}_complete_all_2022_replaced.zarr",),
                    ]
   
    if grid == "global_dense":
        directory = join(c.features_dir,
                "global_dense_grid", 
                "complete",
                "concat",
                "replace_2022",
                )
                
        local_paths = [join(directory, file) for file in listdir(directory) if file.endswith(".zarr")]
        local_paths.sort()

    return local_paths

    
def get_model_weights_from_kfold_results(kfold_results, retrained_final_model_weights=True):   
    if retrained_final_model_weights:
        m = kfold_results.get("prediction_model_weights")
        if m is not None:
            b = kfold_results["prediction_model_intercept"]
            return m, b
        else:
            log_text(("Could not get weights from retrained final model. "
            "Using flattened CV weights instead..."), print_text="warn")

    best_lambda_idx = ir.interpret_kfold_results(
    kfold_results,
    "r2_score",
    )[0][0][0]

    stacked_weights = [fold[0][best_lambda_idx] for fold in kfold_results["models"]]
    m = np.mean(stacked_weights, axis=0) #taking the mean from the folds
    

    stacked_intercepts = [fold[0][best_lambda_idx] for fold in kfold_results["intercepts"]]
    b = np.mean(stacked_intercepts, axis=0) #taking the mean from the folds
    
    
    return m,b 

def get_clip_bounds_from_kfold_results(kfold_results):
    if type(kfold_results) is list:
        error_text = ("clip bounds not saved in data output"
        "ensure that you are predicting from a newly run model or "
        "try running predictions without clipping")

        solver_kwargs = kfold_results[0].get("solver_kwargs")
        if solver_kwargs is None:
            log_text(error_text, print_text="warn")
            raise Exception
        clip_bounds = solver_kwargs["clip_bounds"][0]

        for item in kfold_results:
         assert all(clip_bounds == item["solver_kwargs"]["clip_bounds"][0])

    else:
        solver_kwargs = kfold_results.get("solver_kwargs")
        if solver_kwargs is None:
            log_text(error_text, print_text="warn")
            raise Exception
        clip_bounds = kfold_results["solver_kwargs"]["clip_bounds"][0]

    return clip_bounds


def get_log_transform_added_val(kfold_results, app):
    if type(kfold_results) is not list:
        kfold_results = [kfold_results]
        
    c_app = kfold_results[0]["c_app"]
    
    if c_app["transformation"] != "log":
        return None
    
    if "pre_transform_added_value" in c_app:
        return c_app["pre_transform_added_value"]
    
    ### Now we have to actually calculate this:
    # This section of code is used for backwards compatibility.
    # Previously, we did not save the pre_transform_added_value
    # We have to ge the source labels data unfortunately
    polygon_id = c_app.get("polygon_id_colname")
    
    c,c_app = extract_config(app, from_meta=True)[0:2]
    Y_df = mio.get_Y(c).dropna(subset = ["lon","lat",c_app["colname"]])
    
    
    if polygon_id:
        locs = np.concatenate([np.concatenate(r['locations_test']) for r in kfold_results]).squeeze()
        
        y = Y_df.groupby(polygon_id)[c_app["colname"]].mean().loc[locs]
        
    else:
        locs = np.vstack([np.vstack(r["locations_test"]) for r in kfold_results])
        
        y = Y_df.groupby(["lon","lat"])[c_app["colname"]].mean().loc[list(zip(locs[:,0],locs[:,1]))]
    
    
    ### Now we finally have the Ys. We just need to get val from the transfrom
    
    y_transform, val_added = transforms.custom_log_transform(y)
    
    return val_added

def get_model_weight_data_dict(kfold_results, weight_option = "retrain_model" ):
    """"
    region: bool

    weight_options: "retrain_model", "cv_avg"
    """

    use_retrain_weights = (weight_option == "retrain_model")
    data_dict = {}

    if type(kfold_results) is dict:
        log_text("No regions in results object")
        region_models = False

    elif type(kfold_results) is list:
        log_text("Results object has seperate models for regions")
        region_models = True

    if region_models:
        for item in kfold_results:
            region_name = item["region"]
            region_dict = {}
            region_dict["m"],region_dict["b"]  = get_model_weights_from_kfold_results(item, use_retrain_weights)
            
            data_dict[region_name] = region_dict

    else:
        global_dict = {}
        global_dict["m"],global_dict["b"]  = get_model_weights_from_kfold_results(kfold_results, use_retrain_weights)
        data_dict["global"] = global_dict

    return data_dict


def get_regions_and_region_type_from_data_dict(data_dict):
    if "global" not in data_dict:
        region_models = True

        ## Use keys to figure out if we are in a country or continent model
        ## TODO make this check less messy:
        if list(data_dict.keys())[0] in c.continent_vm_dict.keys():
            region_type = "continent"
        else:
            region_type= "country"

    else:
        region_models = False
        region_type = None

    return region_models, region_type

    
    
def pred_y_from_model_weight_dict(data_dict, X, clip_preds = [None, None], regions = None, 
return_locs_and_preds=True):
    """
    option to provide a region array


    """
    region_models, region_type = get_regions_and_region_type_from_data_dict(data_dict)
    

    if regions is not None:
        if X.shape[1] == 4002: ## If there are 4002 columns, we assume input Xs have lon, lat coords as first two cols
            locs = X[:,:2]
            X = X[:,2:]
            
    
    else:
        #if regions not provided, X should include lon, lat coords
        X_cols = X.shape[1]
        if X_cols == 4002: ## If there are 4002 columns, we assume input Xs have lon, lat coords as first two cols
            locs = X[:,:2]
            X = X[:,2:]
            
        if region_models:
            assert X_cols == 4002, "Regions need to be calculated, but X is missing locs"
            locations = pd.DataFrame(locs)
            locations.columns = ["lon", "lat"]

            regions_df = io.get_region_latlon_intersection(c.shp_file_dict[region_type], 
                c.region_col_dict[region_type], locations,region_type,grid = None )
            regions = regions_df[region_type].to_numpy()
        else:
            regions = np.full( len(X),"global")

    predicted_vals = np.full(len(X), np.nan)

    for region in np.unique(regions):
        skip_counter = 0 
        if region not in data_dict.keys():
            log_text(f"No regional model for {region}")
            log_text("We cannot produce estimates here...")
            skip_counter += 1
            continue 
        
        region_idxs = (regions == region)

        X_region = X[region_idxs,:]

        predicted_vals[region_idxs] = X_region.dot(data_dict[region]["m"]) + data_dict[region]["b"]

    prop_skipped = skip_counter / len(np.unique(regions))
    if prop_skipped > .5:
        log_text("More than 50% of regions skipped", print_text="warn")
    if any(clip_preds):
        predicted_vals = np.clip(predicted_vals, *clip_preds)

    if return_locs_and_preds: #Used for Xb
        return regions.flatten(), np.hstack([locs, predicted_vals.reshape(-1,1), regions.reshape(-1,1)])
    else: #Used for evaluating Ys
        return predicted_vals

def check_kfold_results_uses_ridge(kfold_results):
        # Figure out if we have a hurdle model
    single_result = kfold_results
    if type(kfold_results) is list:
        single_result = kfold_results[0]
    if single_result["c_app"]["solve_function"] == "Ridge":
        return True
    else:
        return False


def load_all_app_data_dict(path = APP_DATA_PATH):
    """
    For running Xb, we want to have a saved file with all the betas from the various apps.
    
    This will allow us to avoid getting model weights over and over again.
    
    """
    if exists(path):
        return pickle.load(open(path, "rb"))
    else:
        return {}

def write_all_app_data_dict(data_dict, path=APP_DATA_PATH):
    """
    For running Xb, we want to have a saved file with all the betas from the various apps.
    
    This will allow us to avoid getting model weights over and over again.
    
    """
    pickle.dump(data_dict, open(path, "wb"))


def get_and_write_data_dictionary(apps, weight_option, overwrite_saved_weights, clip):
    
    all_app_data_dict = load_all_app_data_dict()

    for app in apps:
        app_weights_are_written = (app in all_app_data_dict)

        if app_weights_are_written and not overwrite_saved_weights:
            continue

        else:
            kfold_results = ir.get_kfold_results_from_main(app)
            
            if not check_kfold_results_uses_ridge(kfold_results):
                log_text(f"Skipping {app}. Does not use Ridge model...", print_text="warn")
                continue
            
            data_dict = get_model_weight_data_dict(kfold_results, weight_option=weight_option)

            if clip:
                data_dict["clip_bounds"] = get_clip_bounds_from_kfold_results(kfold_results)
            else:
                data_dict["clip_bounds"] = [None, None]
            
            data_dict["log_value_added"] = get_log_transform_added_val(kfold_results, app)
            
            all_app_data_dict[app] = data_dict
            
            ## Save after each iteration. Will manage data best.
            write_all_app_data_dict(all_app_data_dict)
    
    all_app_data_dict_out = all_app_data_dict.copy()
    for key in all_app_data_dict:
        if key not in apps:
            _ = all_app_data_dict_out.pop(key)

    return all_app_data_dict_out

def save_intermediate_preds(preds, app, X_filename):
    app_dir = join(XB_PRED_DIRECTORY, app)
    if not exists(app_dir):
        mkdir(app_dir)
    
    output_file_name = app + "_" + X_filename.split(".")[0].split("/")[-1] + ".p"
    
    pickle.dump(preds, open(join(app_dir,output_file_name), "wb"))




def Xb_pred(apps, grid = "global_sparse", weight_option = "retrain_model", clip = True, 
overwrite_saved_weights = False, unlog=True, save_outputs = False ):
    """
    Function that produces global tile-level estimates of outcome or outcomes.

    apps: name of label in metadata.py,  WIP: can also be a list of apps

    grid: ["global_sparse", "global_dense"] whether to predict app globally with sparse or dnese grid

    weight_options: ["retrain_model", "cv_avg"] : Whether to retrain a model using all the training data 
        and use those model weights or just take the average weights from the 5 fold cross val models with the best lam
    
    clip: bool : whether to clip the preds using the clipping procedure used in training or not.

    unlog: bool : If model was log transformed, predictions will be log transformed. Bool to undo log transform.

    save_outputs: bool to save xB for each app and X zarr file. Helpful for memory management when running global.
                When True, preds are not returned.
    """
    start_time = time.time()

    # First we deal with grid options
    grid_options = ["global_sparse", "global_dense"]   

    if grid not in grid_options:
        bad_string = ("Grid parameter not acceptable."
        " Must be one of the following: \n" +
        str(grid_options) )
        log_text(bad_string)
        raise NotImplementedError
    
    if type(apps) is str:
        apps = [apps] #make into list if it's a string

    local_paths = get_all_grid_paths(grid)

    all_app_data_dict = get_and_write_data_dictionary(apps, 
    weight_option=weight_option, 
    overwrite_saved_weights=overwrite_saved_weights, 
    clip=clip)

    if not save_outputs:
        stacked_preds_dict = {}
        for app in all_app_data_dict.keys():
            stacked_preds_dict[app] = []

    log_text("Cycling through Xs")
    for file in local_paths:
        log_text(f"Running preds for {file}" )
        X = zarr.load(file)
        if X is None:
            log_text("Zarr file couldn't be loaded. This is probably a permissions issue")
            log_text("fpath = " + file)
            raise Exception
        for i, app in enumerate(all_app_data_dict.keys()):
            log_text(app)
            data_dict = all_app_data_dict[app]
            clip_preds = all_app_data_dict[app]["clip_bounds"]
            log_val_added = all_app_data_dict[app]["log_value_added"]
            _, region_type = get_regions_and_region_type_from_data_dict(data_dict)
            if i == 0:
                regions = None
                prev_region_type = None
            elif prev_region_type != region_type:
                regions = None # If region type changes, re-run get region function
                log_text("prev region type is not current region type!")
            
            regions, preds = pred_y_from_model_weight_dict(data_dict, X, clip_preds=clip_preds, regions=regions)

            ####Unlog transform the preds
            if (log_val_added is not None) and unlog: # We need to undo the log transform
                preds[:,2] = np.exp(preds[:,2].astype(float)) - log_val_added

            prev_region_type = region_type

            if save_outputs:
                _ = save_intermediate_preds(preds, app, file)
                del preds
            else:
                stacked_preds_dict[app].append(preds)
        del X
    
    if not save_outputs:
        for app in stacked_preds_dict.keys():
            stacked_preds_dict[app] = np.vstack(stacked_preds_dict[app])

        elapsed_time = time.time() - start_time

        log_text(f"Time elapsed (mins) = {elapsed_time/60}")

        return stacked_preds_dict
    else:
        return "Done!"


def get_test_X_locations_y(kfold_results, c=c):
    if type(kfold_results) == list:
        kfold_results = kfold_results[0]
    # get config, as saved in the output data 
    try:
        c_app = kfold_results["c_app"]
    except:
        log_text("c_app not in saved output. Test set evaluation function only works for newly updated data outputs",
        print_text="warn")
        raise Exception

    transformation = c_app.get("transformation")
    polygon_id_colname = c_app.get("polygon_id_colname")

    c = io.get_outcomes_filepath(c, c_app)

    X, locations, y, regions = io.get_X_locations_y(c, c_app, 
                                                polygon_id_colname)
    (
    _,
    X_test,
    Y_train, #We need Y_train to do our custom log transform
    Y_test,
    _,
    locations_test,
    _,
    regions_test
    ) = parse.merge_dropna_split_train_test(c, c_app["application"], c_app, c_app["labels_filename"], X,
                                            locations, y, regions)
    if transformation == "log":
        Y_test = transforms.custom_log_val(Y_test, Y_train)
    return X_test, Y_test, locations_test, regions_test


def predict_test_set(app, weight_option = "retrain_model", filepath_override=False, clip=True):

    if filepath_override: # We can get the kfold results by specifing a file path
        kfold_results = pd.read_pickle(filepath_override)

    else:
        kfold_results = ir.get_kfold_results_from_main(app)

    
    if clip:
        clip_preds = get_clip_bounds_from_kfold_results(kfold_results)

    X_test, Y_test, locations_test, regions_test = get_test_X_locations_y(kfold_results)

    check_for_test_contamination(kfold_results,locations_test)

    data_dict = get_model_weight_data_dict(kfold_results, weight_option=weight_option)
    
    y_hat = pred_y_from_model_weight_dict(data_dict, 
    X_test, 
    regions=regions_test,
    clip_preds = clip_preds, 
    return_locs_and_preds=False)
    
    null_idxs = np.isnan(y_hat)
    
    if any(null_idxs):
        log_text( ("Region model was skipped in training and observations were dropped. "
                   "This is probably because there was no variation in the region model"
                   "Observations from the missing region will be dropped from the test set"),
                   print_text="warn")
        
        y_hat = y_hat[~null_idxs]
        Y_test = Y_test[~null_idxs]

    return y_hat, Y_test, r2_score(Y_test,y_hat)


def stringify_lon_lats(locations):
    str_locations = np.core.defchararray.add(locations[:,0].astype(str), "_" )
    str_locations = np.core.defchararray.add(str_locations, locations[:,1].astype(str))
    return str_locations


def check_for_test_contamination(kfold_results, locations_test):
    
    if type(kfold_results) is dict:
        kfold_results = [kfold_results]
    
    if len(locations_test.shape)==2 and locations_test.shape[1] == 2: #locations are n x 2 (tile solve)
        
        test_strings = stringify_lon_lats(locations_test)
        locs = []
        
        for item in kfold_results:
            region_locs = item["locations_test"][0]
            assert region_locs.shape[1] == 2
            locs.append(region_locs)
            
        train_strings = stringify_lon_lats(np.vstack(locs))
    
    else: #we have locations that are n x 1, polygon
        test_strings = locations_test.flatten()
        
        locs = []

        for item in kfold_results:
            region_locs = item["locations_test"][0].flatten()
            locs.append(region_locs)
            train_strings = np.hstack(locs)
            
    contamination = any(np.isin(train_strings, test_strings))
        
    if contamination:
        log_text("There is test set contamination, evaluation is invalid.", print_text="warn")
        raise Exception
        
