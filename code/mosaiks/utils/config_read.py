import os
import time
import importlib
import json
import numpy as np
import pandas as pd

from mosaiks.utils.logging import log_text
from mosaiks.utils import io
from mosaiks.solve import data_parser as parse
from mosaiks import config as c


###
# Functions for converting metadata to pd object and back to dict
def metadata_dict_to_df(metadata):
    df = pd.DataFrame(metadata).T
    df['config'] = list(metadata)
    return df

def metadata_df_to_dict(df):
    j = df.T.to_dict()
    return j

##############

def parse_metadata(app, c=c):
    """
    Extract the metadata you need to run a single job from the metadata file.
    This function is intended to be run as a first step before passing
    to the autotuning functions.
    """
    assert app in metadata.keys(), "app not found in metadata keys."
    
    # the row containing the information we want
    row = metadata[app]
    outcome_name = row["col_name"]
    labels_directory = row["labels_directory"]
    labels_file = row["labels_filename"]
    grid = row["grid_name"]
    region_type = row["region_type"]

    polygon_id_colname = row["polygon_agg_col_name"]
    shp_file_name = row["shp_file_name"]
    save_X_name = row["save_X_name"]
    
    # get the model type and determine the solve function
    model_type = row["model_type"]
    if model_type == "continuous":
        solve_function = "Ridge"
    elif model_type == "classifier":
        solve_function = "OVR_classifier"
    else:
        raise NotImplementedError("model_type not recognized.")

    # build c_app dictionary from the above
    c_app = {
        "application": app,
        "colname": outcome_name,
        "grid": grid,
        "solve_function": solve_function,
        "labels_filename": labels_file,
        "labels_directory": labels_directory,
        "polygon_id_colname": polygon_id_colname,
        "shp_file_name": shp_file_name,
        # "units": units, # see comment above about units
        "save_X_name": save_X_name,
        "region_type": region_type
    }
    
    return c_app


def extract_c_app(c_app):
    """
    Helper function to extract commonly used keys (outcome_name, labels_file,
    grid, polygon_id_colname) from a c_app dictionary returned by 
    `parse_metadata()`.
    """
    ret = ()
    for key in ["colname", "labels_filename", "grid",
                "polygon_id_colname"]:
        ret += (c_app[key],)

    return ret


def extract_config(app, c=c, from_meta=False,auto_tune=False,
                   load=False, subdirectory = False):
    """
    Extract the information you need from your config.

    Args:
        app: string. The name of your config (must have a config of the same name in
            in `mosaiks/configs`.)
        load: boolean. If past predictions exist, should they be
            loaded and returned? Default is `False`.
        from_meta: boolean. Load the metadata from the CSV instead of from a config?
            Note that this loads an incomplete config. Best used with autotuning.
        autotune: boolean. Are you autotuning? Saves to autotuning subfolder.
        sub_directory: str or False The name of the subdirectory to read a config
    """
    if from_meta:
        # Most parsing is already implemented in other functions.
        c_app = parse_metadata(app)
        
        (outcome_name, labels_file, grid, 
         polygon_id_colname) = extract_c_app(c_app)
        
        c = io.get_outcomes_filepath(c, c_app)
    else:
        # Load task-specific config as module
        if subdirectory:
            log_text("getting config from subdirectory")
            task_config = f"mosaiks.configs.{subdirectory}.{app}"
            # The file gets executed upon import, as expected.
            label_cfg = importlib.import_module(task_config)
            
            
        else:
            log_text("getting config from main directory")
                          
            task_config = "mosaiks.configs." + app
            # The file gets executed upon import, as expected.
            label_cfg = importlib.import_module(task_config)
            
            

        # Get attributes from task-specific config file
        c_app = getattr(label_cfg, app)
        if subdirectory:
            c_app["save_folder"] = subdirectory

        outcome_name = c_app["colname"]

        # Get filepath for labels
        labels_file = c_app["labels_filename"]

        #Choose grid
        grid = c_app["grid"]

        polygon_id_colname = c_app.get("polygon_id_colname", None)


    
  # Obtain the file paths you need for your application and feature type
    # (this simply calls the filepaths listed in config.py for location of 
    # labeled and feature datasets)
    c = io.get_filepaths(c, app, c_app, labels_file, auto_tune=auto_tune)
    c = io.get_filename(c, app, c_app, labels_file)
    
    return (c, c_app, outcome_name, labels_file, grid, polygon_id_colname)
    


def prep_data(c, app, c_app, polygon_id_colname,
             labels_file):
    """
    Load features and merge them with labels. Split into train and test.
    
    """


    X, locations, y, regions = io.get_X_locations_y(c, c_app,
                                           polygon_id_colname,
                                           )

    (X_train, _, Y_train, _, locations_train, _, regions_train, _
    ) = parse.merge_dropna_split_train_test(c, app, c_app,
                                                labels_file, X, locations,
                                                y, regions)

    return X_train, Y_train, locations_train, regions_train