import io
from os.path import join, exists
#from os import chmod, mkdir, chown
#import grp
import copy

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.vectorized import contains
from shapely.geometry import Point

import dask
import dask.array as da
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
import gc

from mosaiks.utils.logging import log_text as log_text, robust_chmod

import warnings

import zarr


#import grp
def create_dir(dirName):
    '''
    Helper function to create directories automatically
    '''
    
    if not exists(dirName):
        mkdir(dirName)

        log_text(f"Directory {dirName} Created ")
        #robust_chmod(dirName, 0o774)

        #change directory group to maps100 
        #chown(dirName, -1, grp.getgrnam("maps100").gr_gid)



def get_suffix(c, appname, c_app):
    """Return the standard filename suffix that we are using to
    keep track of different model settings.
    Args:
        c (config module): The module you get when importing config.py
        appname (str): The name of the application (e.g. housing)
    Returns:
        str: The sub-suffix representing grid-specific settings
        str: The sub-suffix representing sample-specific settings.
            grid_str + '_' + sample_str is the full suffix you will
            need, but sometimes you need them separately as well.
    """

    c_grid = c.grid
    c_smpl = c.sampling
    c_img = c.images
    c_app = c_app

    grid_str = "{}_{}_{}".format(c_grid["area"], c_img["zoom_level"], c_img["n_pixels"])
    sample_str = "{}_{}_{}".format(
        c_app["sampling"], c_smpl["n_samples"], c_smpl["seed"]
    )

    return grid_str, sample_str


def get_outcomes_filepath(c, c_app):
    """
    Helper function for getting only the filepath to the outcomes, as opposed to
    all the filepaths you need for reading/writing a task, as is implemented in 
    `get_filepaths`.
    """
    c.outcomes_fpath = join(
        c.data_dir,
        "int",
        "applications",
        c_app["labels_directory"],
        "{}".format(c_app["labels_filename"]),
    )

    return c


def get_filename(c, app, c_app, labels_file):
    """
    Helper function for updating the config to get filenameing convention.
    Convention includes the following components in the following order:
    - label name
    - class/reg/hurdle
    - pool/continet
    - log/level
    - observedbounds/intuitivebounds/combobound/nobound
    - intercept/nointercept
    - tile/poly
    - popweight/areaweight
    """
    if c_app.get("solve_function") == "OVR_classifier":
        class_or_reg = "class"

    elif c_app.get("solve_function") == "Hurdle":
        class_or_reg = 'hurdle'
    
    elif c_app.get("solve_function") == "new_Hurdle":
        class_or_reg = 'newHurdle'
    else:
        class_or_reg = "reg"
    
    transformation = c_app.get("transformation")
    if transformation is None:
        transformation = "lvl"
    
    if c_app.get("polygon_id_colname") is None:
        aggregation = "tile"
    else:
        aggregation = "poly"
        
    if c_app.get('intercept'):
        intercept = 'intercept'
    else:
        intercept = 'nointercept'
    
    
    # no bound
    if c_app.get('bounds_pred') in [[None,None], None]:
        if c_app.get('bound_replace_None_w_observed'):
            bounds = 'observedbounds'
        else:
            bounds = 'nobound'
    elif type(c_app.get('bounds_pred') ==list):
        # intuitive bounds with none replaced
        if c_app.get('bound_replace_None_w_observed') and (None in c_app.get('bounds_pred')):
            bounds = 'combobound'  
        else:
            bounds = 'intuitivebounds'
    else: #when bounds_pred is auto
        bounds = 'observedbounds'

        
    if c_app.get('pop_weight_features'):
        weight = 'popweight'
    elif c_app.get("polygon_id_colname") and c_app.get('pop_weight_features') in [None, False]:
        weight = 'areaweight'
        
    # This is only meant to be from the auto tuning pipeline and not user set
    if c_app.get('region_type'):
        continents = c_app['region_type']
    else:
        continents = 'pool'
    
    if c_app.get("polygon_id_colname"):
        c.fname = "_".join(
            [app, class_or_reg, continents, transformation, bounds, intercept, aggregation, weight]
        )
    else:
        c.fname = "_".join(
            [app, class_or_reg, continents, transformation, bounds, intercept, aggregation]
        )

    c.fname = c.fname.replace("__","_")
    
    return c

def get_filepaths(c, app, c_app, labels_file, auto_tune=False):
    """
    Helper function for updating the config with all the filepaths you need to
    read and write data for a task.
    """
    label_directory = c_app.get("labels_directory", app) #if labels directory not set, assume label directory is the app name
    c.outcomes_fpath = join(
        c.data_dir,
        "int",
        "applications",
        label_directory,
        "{}".format(labels_file),
    )
    
    # define paths where outputs will be saved
    # if the task config is in a subdirectory, the saving path will be for
    # the "non-main" version, and will accordingly be put in a subdirectory
    # of the outputs folder.
    # If the task config is not in a subdirectory, the saving path will be for
    # the "main" version, and will be saved in the outputs folder directly.
    
    save_folder = c_app.get("save_folder")
    
    if auto_tune:
        create_dir(c.main_plot_dir.replace("main","") + app)
        c.plot_savedir = c.main_plot_dir.replace("main","") + app + "/autotune"
        create_dir(c.plot_savedir)
        create_dir(c.main_pred_dir.replace("main","") + app)
        c.pred_savedir = c.main_pred_dir.replace("main","") + app + "/autotune"
        create_dir(c.pred_savedir)
        log_text('Setting saving directories for plots and preds to ' +
            app + '/autotune. Outputs will be treated as non-main.')
        app_name = app
    else:
        if save_folder:
            c.plot_savedir = c.main_plot_dir.replace("main","") + save_folder
            create_dir(c.plot_savedir)
            c.pred_savedir = c.main_pred_dir.replace("main","") + save_folder
            create_dir(c.pred_savedir)
            log_text('Setting saving directories for plots and preds to sub-folder ' +
                save_folder + '. Outputs will be treated as non-main.')
            app_name = save_folder
        else:

            create_dir(c.main_plot_dir.replace("main", app) )
            c.plot_savedir = c.main_plot_dir.replace("main", app)

            create_dir(c.main_pred_dir.replace("main", app) )
            c.pred_savedir = c.main_pred_dir.replace("main",app )


            log_text('Setting saving directories for plots and preds to /app.')
            app_name = app

    # cretae logs folder
    c.log_savedir = c.main_log_dir.replace("main","") + app
    create_dir(c.log_savedir)
        
    return c



# def subset_zarr_to_y(zarr_path, Y_df): 
#     """Lazily load zarr files and subset features to only coordinates from label df
#     Args:
#         zarr_path (str): Path to zarr grid file containing features
#         Y_df (df): labels df - must contain both 'lat' & 'lon' columns
#     Returns:
#         X_sub (df): Subsetted df of features from label coordinates
#     """
#     log_text("Loading labels")
#     labels = Y_df.loc[:,['lat','lon']] #get latlons of Y
#     labels['id'] = labels['lon'].astype("str") + "_" + labels['lat'].astype("str") #create id column
    
#     arrs = da.from_zarr(zarr_path)
    

#     log_text("Converting features to dask array")
#     X = dd.from_dask_array(arrs,
#                             columns=["lon", "lat"] + ["X_" + str(i) for i in range(arrs[:,2:].shape[1])]) #create dask df from array
           
#     X["lon"] = X["lon"].round(3) # deal with potential for float precision issue after converting to float
#     X["lat"] = X["lat"].round(3) 
            
#     #create id column
#     X['id'] = X['lon'].astype("str") + "_" + X['lat'].astype("str")

#     #subset features to labels' latlon and compute
#     log_text("Subsetting features to label coordinates. This might take several mins, see Dask Client Dashboard for progress")
#     X_sub = X.loc[X['id'].isin(labels['id'])].compute()
    
#     log_text ("Subsetting done.")

#     return X_sub

def subset_zarr_to_y(zarr_path,Y_df):
    
    log_text("Subsetting features to label coordinates")
    arrs = zarr.load(zarr_path)
    X_lons = arrs[:,0]
    subset = np.isin(Y_df["lon"], X_lons)
    Y_df = Y_df.loc[subset]
   
    z_df = pd.DataFrame(arrs)
    
    z_df.columns = ["lon", "lat"] + ["X_" + str(i) for i in range(arrs[:,2:].shape[1])]
    
    X_sub = Y_df[["lon","lat"]].merge(z_df, how="inner", on=["lon","lat"])
    
    X_sub["id"] = X_sub["lon"].astype(str) + "_" + X_sub["lat"].astype(str)
    
    return X_sub




def weighted_groupby(df, groupby_col_name, weights_col_name, weight_sum_colname = None, cols_to_agg="ALL COLUMNS"):
    """
    Applies a weighted groupby to a dataframe where the weights are given in a dataframe column.
    
    Currently, this is used to area or population weight a large dataframe of features.
    
    weight_sum_colname is an optional parameter. When included, there is an additional column in the output that gives the sum of the weights for that polygon or groupby item. This allows for chunking and future re-weighting.
    
     Parameters
    ----------
    df : pd.DataFrame
        dataframe with that will receive a weighted grouping
    groupby_col_name : str or list of strings
        Will be passed as the first arg to pd.DataFrame.groupby function
    weights_col_name: str
        The column name that contains the weights for aggregating the dataframe
    weight_sum_colname : (optional) str
        If included, the weights will be saved in an output column. This string is the column name.
    cols_to_agg : (optional) list of column names
        List of columns that will receive the weighting and will be output. Default is all columns other than the weighting column and the groupby column name.
        
    Returns
    -------
    out : pd.DataFrame
    """
    df_cols = list(df.columns)
    
    if type(groupby_col_name) is str:
        assert groupby_col_name in df_cols
    else:
        assert all([group_col in df_cols for group_col in groupby_col_name])
    assert weights_col_name in df_cols
    
    if cols_to_agg == "ALL COLUMNS":
        cols_to_agg = df_cols
        cols_to_agg.remove(groupby_col_name)
        cols_to_agg.remove(weights_col_name)
    
    if len(df) < 1: #if df is blank, return blank frame with expected colnames
        print("df < 1 ... returning blank dataframe")
        if weight_sum_colname:
            cols_to_agg += [weight_sum_colname]
            df[weight_sum_colname] = []

        return df[cols_to_agg]
        
    else:
        for col in cols_to_agg:
            assert col in df_cols
            
    def weighted(x, cols, weights=weights_col_name):
        return pd.Series(np.average(x[cols], weights=x[weights], axis=0), cols) 
    g = df.groupby(groupby_col_name)
    out = g.apply(weighted, cols_to_agg)
    
#     print("\n")
#     print("First stage weight results")
#     print(out.head())
    if weight_sum_colname:
        
        sums = g[weights_col_name].sum().rename( weight_sum_colname)
#         print("count col group results")
#         print(sums.head)
        out = pd.concat([sums,out], axis=1)
    
    return out




def agg_polygon_X_y(c, c_app, X, Y_df, polygon_id_colname, weight_sum_col= False):
    
    """ 
     This function is in progress. It will be used to aggregate the X matrix at the polygon level. It will be called by 
     get_X_locations_y()
     
     If weight_sum_col, then a count column is returned with X. This can be used for two stage weighting.
     
     
    """
    y_colname = c_app["colname"]
     
    pop_weight = c_app.get("pop_weight_features")
    
    Y_df = Y_df[[y_colname, polygon_id_colname, "lat","lon"]]
    
    if pop_weight:
        if c_app["grid"] == "global_sparse":
            pop_density_df = pd.read_csv(c.sparse_grid_pop_density_path)
        else:
            pop_density_df = pd.read_pickle(c.dense_grid_pop_density_path)
            
        pop_density_df = pop_density_df.dropna()[[c.pop_density_val_colname, "lon", "lat"]] #can't use NAs, they will be given 0 weight
        pop_density_df["lon"] = np.round(pop_density_df.lon,3) #Always want to do this when merging on floats, just in case
        pop_density_df["lat"] = np.round(pop_density_df.lat,3)
        
        if pop_weight == "IHS": # Using log with no transformation results in negative weights
            pop_density_df[c.pop_density_val_colname] = np.arcsinh(pop_density_df[c.pop_density_val_colname])
        
        if pop_weight == "log+1":
            pop_density_df[c.pop_density_val_colname] = np.log(pop_density_df[c.pop_density_val_colname]+1)
        
        if pop_weight == "log+100":
            pop_density_df[c.pop_density_val_colname] = np.log(pop_density_df[c.pop_density_val_colname]+100)
             
        log_text("Merging pop density values with label data frame")
        Y_df = Y_df.merge(pop_density_df, 
                          on = ["lat", "lon"],
                          how = "left")

        # If pop weight dataframe is missing some values, fill with the min of the observed weights
        # TODO if we add other kinds of weighting, this may not be an appropriate procedure
        pop_density_min_value = np.min(pop_density_df[c.pop_density_val_colname])
        Y_df.loc[Y_df[c.pop_density_val_colname].isnull(), c.pop_density_val_colname] = pop_density_min_value



    log_text("Merging X and Y; can be slow on dense grid")
    X = Y_df.merge(X, # # Inner join so nonmatching labels and Xs are dropped
              on=["lat","lon"],
              how='inner'
       ).drop(columns=["lon","id"])
    
    lat = X.pop("lat")
    
    
    if pop_weight:
        weight_val_colname = c.pop_density_val_colname
    
    
    else:
        log_text("This implementation of polygon aggregation adjusts area based on cos(lat). Not in 100Maps.")
        weight_val_colname = "cos_lat"
        
        # Take cos(lat). Note that Lat here is a named index in X dataframe
        X[weight_val_colname] = np.cos(np.deg2rad(lat))
        
    total_pop_weights_colname = None
    if weight_sum_col:
        total_weights_colname = "weight_sum"
        
    X = weighted_groupby(X, polygon_id_colname, weight_val_colname, total_weights_colname)

    y = X.pop(y_colname)

    return X, y


def get_min_distance_region(lat, lon, shp, region_col):
    """
    Helper function which returns the nearest region to a given point.
    Used to assign regions to lat/lon points that do not fall within a region
    polygon.

    Parameters
    ----------
    lat: (numeric) the latitude of the point.
    lon: (numeric) the longitude of the point.
    shp: (GeoPandasDataFrame) the shapefile contianing the regions
    region_col: (str) the name of the column containing the region ID.
    """
    point = Point(lon, lat)

    # Note this raises a  CRA warning, but the resulting continent assignments appear
    # to be correct. This is likely because the approximation error due to
    # flattening the globe is small enough to not matter when it comes
    # to assigning regions. We have looked at the output and confirmed that
    # things look reasonable. So we suppress the warning.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        distances = shp.distance(point)
    
    region = shp.loc[distances.idxmin(), region_col]
    
    return region


def get_region_latlon_intersection(shp_path, region_col, locations, region_type, grid):
    """
    Given a shapefile containing regions, get a mapping of latlons to each region,
    and save the mapping as a csv to be used to subset features.
    
    Parameters
    ----------
    shp_path: str.
        The path to the shapefile.
    region_col: str.
        The name of the column which contains the region ID.
    locations: DataFrame.
        DataFrame of the locations you want to take the intersection with.
    grid parameter: if is == "US_dense" skip search
    """
    if grid == "US_dense":
        regions = copy.deepcopy(locations[["lat", "lon"]])
        regions[region_type] = "North America"
        return regions
    
    input_idxs = locations.index
    log_text("Hit get_region function")
    shp = gpd.read_file(shp_path)
    ncol = locations.shape[1]
    # creates one column per region with a boolean for whether each point is
    # inside that region 
    locations = locations.assign(
        **{row[region_col]: contains(row["geometry"],
                                      locations.lon,
                                      locations.lat)
                    for i, row in shp.iterrows()
          }
    )
 
    # add a column with the region name by identifying the column with the
    locations[region_type] = locations.iloc[:, ncol:].idxmax(axis=1)

    # in some cases, the region columns may be False for all regions, in which
    # case idxmax returns the first column. We assign these regions to the
    # nearest continent.
    missing_regions = ~(locations.iloc[:, ncol:].max(axis=1))
    
    if missing_regions.sum() > 0:
        missing_regions_fixed = locations.loc[missing_regions,].apply(
            lambda x: get_min_distance_region(x.loc['lat'], x.loc['lon'], shp,
                                            region_col), axis=1)

        locations.loc[missing_regions_fixed.index, region_type] = missing_regions_fixed.values
    
    # return only the lat, lon, and region name

    return locations[["lat", "lon", region_type]].loc[input_idxs]



def get_Y(c):
    """
    Load one or more ground truth variables from a single application.
    This function should only be called on tile solve, so locations = latlon_df.
    
    Parameters
    ----------
    c: python module.
        The config module.
    region: str.
        The name of the region to subset to.
    """

    fpath = c.outcomes_fpath
    
    if fpath[-4:]==(".csv"): 
        Y = pd.read_csv(fpath, low_memory=False)
    elif fpath[-4:]==(".pkl") or fpath[-2:]== (".p"):
        Y = pd.read_pickle(c.outcomes_fpath)
    else:
        raise ValueError("Extension must be .csv, .p or .pkl")
        
    Y["lon"] = Y["lon"].round(3)
    Y["lat"] = Y["lat"].round(3)
        
    Y = Y.sort_values(["lat", "lon"], ascending=[False, True])   

    N = Y.shape[0]
    
    return Y


def clean_regions_df_and_return_regions_series(regions_df, locations, Y_df,
                                               c_app, polygon_agg):
    """
    This function is modeled on `clean_Y_df_and_return_Y_series`.
    It takes a `regions` DataFrame, aggregates to polygons if needed, and
    return a Series.
    """
    regions_name = c_app["region_type"]
    
    if polygon_agg:
        polygon_id_colname = c_app["polygon_id_colname"]
        # get the polygons into the regions df
        regions_df = regions_df.merge(Y_df[["lat", "lon", polygon_id_colname]],
                                      on=["lat", "lon"],
                                      how="right")
        # reduce the regions df to include only one obs per polygon
        # In some cases, a polygon spans multiple continents. 
        # We assign the polygon to the continent it has the most overlap with.
        # first get # of lat lons in each polygon
        regions_df_grouped = regions_df.groupby(
            by=[regions_name, polygon_id_colname]
            ).count().reset_index()

        # assign region by most lat lons
        regions_df = (regions_df_grouped.groupby(polygon_id_colname)
            .max()
            .reset_index()
            .drop(["lat", "lon"], axis=1)
            )
        regions_df.index = regions_df[polygon_id_colname]
        regions = regions_df[regions_name]
    else:
        regions_df['id'] = (regions_df['lon'].astype("str") + "_" +
                            regions_df['lat'].astype("str"))
        regions_df = regions_df.merge(locations, on=["lat", "lon"], how="inner")
        regions = regions_df.set_index('id')[regions_name]

    return regions

def clean_Y_df_and_return_Y_series(Y_df, locations, c_app, polygon_agg):
    if polygon_agg:
        polygon_id_colname = c_app["polygon_id_colname"]            
        Y = Y_df.dropna(subset = [c_app["colname"]]).groupby(polygon_id_colname).first()[c_app["colname"]]
        Y = Y.loc[Y.index.isin(locations.index)]
        # Taking the first value after groupby, all values should be the same since we
        # observe at the polygon level
    else:
        Y_df['id'] = Y_df['lon'].astype("str") + "_" + Y_df['lat'].astype("str") 
        Y_df = Y_df.merge(locations, 
                on=["lat","lon"], 
                how='inner') #get rid of nans in Y latlons
        Y = Y_df.set_index('id')[c_app["colname"]] #only return label column
        
    return Y

def find_vms_and_vm_slices(Y_df):

    """
    Determine VM slices to cycle through based on label's max/min longitude bounds
    
    Parameters
    ----------
    Y_df: n x 2 Dataframe.
        must contain 'lat' and 'lon' columns
    """

    # VM bounds (in order 1-10):
    min_lons = [-180, -102.495, -70.995, -44.495, 13.205, 29.705, 49.305, 78.605, 103.605, 127.105]
    max_lons = [-102.495, -70.995, -44.495, 13.205, 29.705, 49.305, 78.605, 103.605, 127.105, 180.005]

    master_slices = []

    #find max and min lon based on Label
    max_lon = Y_df['lon'].max()
    min_lon = Y_df['lon'].min()

    log_text("min lon of label: " + str(min_lon))
    log_text("max lon of label: " + str(max_lon))

    #determine VMs to use
    log_text ("Subsetting VMs based on label bounds")
    for v in range(1,11):
            
        b = v-1 
            
        #split VM into 10 slices  
        slices = np.linspace(min_lons[b],max_lons[b],11)

        # create masterlist of all slices
        master_slices.append(slices) 
            
        # if label's bounds fall into VM's bounds, add to vmlist
        if min_lons[b] <= min_lon < max_lons[b]:
            log_text("min VM = " + str(v))
            min_vm = v

        if min_lons[b] < max_lon <= max_lons[b]:
            log_text("max VM = " + str(v))
            max_vm = v

                
    vmlist = list(range(min_vm, max_vm+1))
    vmlist = [str(i) for i in vmlist] #convert to str
    log_text("subsetted to the VMs {}".format(vmlist))
    
    # Within each VM, determine which slices to use
    slice_bounds = []
    filelist = []
    vmlist = [int(i) for i in vmlist] # need list of int instead of str

    #find label's new min/max based on subsetted VMs
    reg_labels = Y_df.loc[
                (Y_df['lon'] >= min_lons[(min(vmlist)-1)]) & 
                (Y_df['lon'] < max_lons[(max(vmlist)-1)])
                ]
    reg_max_lon = reg_labels['lon'].max()
    reg_min_lon = reg_labels['lon'].min()
            
    for vm in vmlist:
        for s in range(0,10):
            # create master filelist based on VMs we know we need
            filelist.append('vm' + str(vm) + "_s" + str(s))

            # if label's bounds fall into slice bounds, add to list
            if master_slices[vm-1][s] <= reg_min_lon < master_slices[vm-1][s + 1]:
                log_text("Min slice = vm" + str(vm) + "_s" + str(s))
                slice_bounds.append('vm' + str(vm) + "_s" + str(s))
            
            if master_slices[vm-1][s] <= reg_max_lon < master_slices[vm-1][s + 1]:
                log_text("Max slice = vm" + str(vm) + "_s" + str(s))
                slice_bounds.append('vm' + str(vm) + "_s" + str(s))

    min_index = filelist.index(slice_bounds[0])
    max_index = filelist.index(slice_bounds[1])
   
    slicelist = filelist[min_index:max_index + 1]
    
    return slicelist
    

def cycle_through_X_paths_and_merge_y(
    c, c_app, X_path_list, Y_df, polygon_id_colname, regions_df=None
    ):
    """ Helper function to make get_X_locations_y() less huge. Called by that function.
    """
    X_sub = []
    save_weights_for_final_weighted_agg = False
    regions_cond = (regions_df is not None)

    if (len(X_path_list) > 1) and (polygon_id_colname):
        save_weights_for_final_weighted_agg = True
    
    if type(X_path_list) != list:
        log_text("Warning: Coercing X_path_list to list", print_text='warn')
        X_path_list = [X_path_list]
    
    for local_path in X_path_list:
        log_text("Loading features for" + local_path)
        #subset features to label coordinates
        X_sub_arrs = subset_zarr_to_y(local_path, Y_df)

        log_text("Dropping duplicates if any")
        X_sub_arrs = X_sub_arrs.drop_duplicates(subset = ['id'], keep='first') #id wont be a column with polygon agg

        if polygon_id_colname:
            X_sub_arrs, _ = agg_polygon_X_y(c, c_app, X_sub_arrs, Y_df, polygon_id_colname, save_weights_for_final_weighted_agg)

        #append df to a list, skip if length 0
        if len(X_sub_arrs) > 0 :
            X_sub.append(X_sub_arrs)

    if len(X_path_list) > 1:
        if len(X_sub) == 0:
            raise Exception("The label and the grid (Xs) appear to have no intersection")
        log_text("Concatenating dfs ")
        X_sub_df = pd.concat(X_sub, axis=0)

    else:
        X_sub_df = X_sub_arrs

        
    del X_sub_arrs
    del X_sub
    gc.collect()
    

    if save_weights_for_final_weighted_agg:
        log_text("Final aggregation of X to the polygon level..")
        X_sub_df = weighted_groupby(X_sub_df.reset_index(), polygon_id_colname, "weight_sum")
        
    if polygon_id_colname:
        locations = X_sub_df.index.to_frame()
        Y = clean_Y_df_and_return_Y_series(Y_df, locations, c_app, polygon_agg=True)
        if regions_cond:
            regions = clean_regions_df_and_return_regions_series(regions_df,
                                                                 locations,
                                                                 Y_df,
                                                                 c_app,
                                                                 polygon_agg=True)


    else:
        log_text("Final dropping of duplicates for tile solve, if any")
        X_sub_df = X_sub_df.drop_duplicates(subset = ['id'], keep='first') #id wont be a column with polygon agg

        log_text("No aggregation to polygon. Setting indices...this might take about 1 min")
        X_sub_df = X_sub_df.set_index("id")

        locations = X_sub_df[['lon', 'lat']]
        X_sub_df = X_sub_df.drop(columns=['lon', 'lat'])

        Y = clean_Y_df_and_return_Y_series(Y_df,
                                           locations,
                                           c_app,
                                           polygon_agg=False)
        if regions_cond:
            regions = clean_regions_df_and_return_regions_series(regions_df,
                                                                 locations,
                                                                 Y_df,
                                                                 c_app,
                                                                 polygon_agg=False)
    
    if not regions_cond:
        regions = None

    return X_sub_df, locations, Y, regions


def get_X_locations_y(c, c_app, polygon_id_colname=None, feature_filetype="zarr"):
    """Get RCF features matrices, locations, labels.
    
    Parameters
    ----------
    c: :module:`mosaiks.config`
        Config object
    c_app: dict
        Dicionary of configuration parameters  
    polygon_id_colname: None or str
        column name that includes a unique polygon identifier, used in pd.groupby()
    feature_filetype: "zarr" we may accept ["csv", "npy", "pkl"] in the future
        What file type are the features contained in?

    Returns
    -------
    X : :class:`pandas.DataFrame`
        n x 4000 array of features, indexed by ID
    locations :class:`pandas.DataFrame`
        When NOT aggregating to polygons:
            n x 2 array of longitudes and latitudes, indexed by ID
        When aggregating to polygons:
            n x 1 array of polygon unique identifiers, indexed by ID
        
    Y :class:`pandas.Series`
        n x 1 array of ground truth labels, indexed by ID
    Regions : class: None or `pandas.Series`
        n x 1 array of region identifiers that correspond to Y; or None if c_app["run_continents"] == False
    """

    # Load the feature matrix locally
    
    #Note on filetypes, only zarr implemented
    if feature_filetype != "zarr":
        raise NotImplementedError("Feature input filetype is not currently supported. Currently only .zarr files are supported in the 100Maps pipeline. \n Functionality to read other filetypes may be built in the future.")
    
    #First check grid inputs
    grid = c_app["grid"]
    grid_options = ["global_sparse", "US_dense", "DHS_dense", "global_dense"]
    if grid not in grid_options:
        raise NotImplementedError("Grid parameter not acceptable. Must be one of the following: \n" + str(grid_options) )
    
    #load labels
    Y_df = get_Y(c)

    # get the relevant key
    regions_cond = c_app.get("region_type", None) 

    save_X_name = c_app.get("save_X_name")
    
    save_X_path_exists = False
    # if save X_name is specified, we try to skip all the X calculation steps
    if save_X_name:
        
        pop_weight = c_app.get("pop_weight_features")
        
        # automatically make new save X name for pop weighted features of same outcome
        if pop_weight:
            appendix = "_" + "pop_weight=" + str(pop_weight)
            save_X_name = save_X_name + appendix
    
        save_X_path = join(c.features_dir, 
                           "pipeline_generated_Xs",
                           save_X_name + ".p")
        
        save_X_path_exists = exists(save_X_path)

    if save_X_path_exists:
        log_text("save grid name exists...getting data should be quick")
        X = pd.read_pickle(save_X_path)
        if polygon_id_colname:
            locations  = X.pop(polygon_id_colname)

            Y = clean_Y_df_and_return_Y_series(Y_df,
                                               locations,
                                               c_app,
                                               polygon_agg = True)

            if regions_cond in X.columns and regions_cond is not None:
                regions = X.pop(regions_cond)

            elif regions_cond:
                log_text(f"run regions is specified with region type {regions_cond}, " +
                         f"but there is no {regions_cond} column in saved X")
                raise Exception
            else:
                regions = None
        
        else:
            locations = X[["lon","lat"]]
            X = X.drop(columns=['lon', 'lat'])
            Y = clean_Y_df_and_return_Y_series(Y_df,
                                               locations,
                                               c_app,
                                               polygon_agg = False)
            
            if regions_cond in X.columns:
                regions = X.pop(regions_cond)

            elif regions_cond:
                log_text("run regions is specified, but there is no `region` column in saved X")
                log_text("Outcome is not polygon aggregated, so we will get the regions using lon, lat locations")
                regions = get_region_latlon_intersection(
                                            shp_path=c.shp_file_dict[regions_cond],
                                            region_col=c.region_col_dict[regions_cond],
                                            locations=Y_df[["lat", "lon"]],
                                            region_type=regions_cond, grid=grid)

                if len(regions) != len(locations):
                    log_text("Regions and locations have different lengths")
                    raise Exception
                
            else:
                regions = None

        for region_option in list(c.shp_file_dict.keys()) + ["regions"]:
            if region_option in X.columns:
                _ = X.pop(region_option)

        return X, locations, Y, regions
    
    ### If we don't load X and `regions_cond` is True, then get regions from `Y_df`
    if regions_cond:
        regions = get_region_latlon_intersection(shp_path=c.shp_file_dict[regions_cond],
                                                region_col=c.region_col_dict[regions_cond],
                                                locations=Y_df[["lat", "lon"]],
                                                region_type=regions_cond, grid=grid)
        if len(regions) != len(Y_df):
            log_text("Regions and locations have different lenghts")
            raise Exception
    else:
        regions = None

    if grid == "global_sparse":  
        log_text("Global Sparse Grid - Patchsize 4 + 6")  
        local_paths = [join(c.features_dir, "global_sparse_grid", f"{grid}_complete_all_2022_replaced.zarr",),
                      ]


    elif grid == "DHS_dense":       
        log_text("DHS Dense Grid - Patchsize 4 + 6")   
        
        prefix = join(c.features_dir, "global_dense_grid", "final", "DHS", "replace_2022")
        local_paths = [join(prefix, f"{grid}_s1.zarr"),
                      join(prefix, f"{grid}_s2.zarr"),
                      join(prefix, f"{grid}_s3.zarr"),
                      join(prefix, f"{grid}_s4.zarr"),
                      join(prefix, f"{grid}_s5.zarr"),
                      join(prefix, f"{grid}_s6.zarr"),
                      join(prefix, f"{grid}_s7.zarr"),]

    elif grid  == "US_dense": 
        log_text("US Dense Grid - Patchsize 4 + 6")
        prefix = join(c.features_dir, "global_dense_grid", "final", "US", "replace_2022")

        local_paths = [join(prefix, f"{grid}_v1_s1.zarr"),
                      join(prefix, f"{grid}_v1_s2.zarr"),
                      join(prefix, f"{grid}_v1_s3.zarr"),
                      join(prefix, f"{grid}_v1_s4.zarr"),
                      join(prefix, f"{grid}_v1_s5.zarr"),
                      join(prefix, f"{grid}_v2_s1.zarr"),
                      join(prefix, f"{grid}_v2_s2.zarr"),
                      join(prefix, f"{grid}_v2_s3.zarr"),
                      join(prefix, f"{grid}_v2_s4.zarr"),
                      join(prefix, f"{grid}_v2_s5.zarr"),
                      join(prefix, f"{grid}_v3_s1.zarr"),]

    elif grid  == "ca_dense_grid": 
        log_text("CA Dense grid - Patchsize 4 + 6")
        local_paths = [join(
            c.features_dir,
            "global_dense_grid",
            f"{c_app['grid']}.zarr",)]
    
    elif grid == "global_dense":
        log_text("Global dense grid - determining VM subfiles to cycle through ")
         
        vmlist = find_vms_and_vm_slices(Y_df[["lat", "lon"]])
        log_text(str(vmlist))
         
        local_paths = []
        for vm_slice in vmlist:
            local_path = join(c.features_dir,
                "global_dense_grid", 
                "complete",
                "concat",
                "replace_2022",
                f"{vm_slice}.zarr",
                )
                
            local_paths.append(local_path)
    
    ### Now this function can take as many broken up zarr file paths as we want
    X, locations, Y, regions = cycle_through_X_paths_and_merge_y(
        c, c_app, local_paths, Y_df, polygon_id_colname, regions
    )
    log_text("Done!")
    
    if regions is not None: #fixes misaligned indices, problematic when concatenating below. Better fix could be added earlier in pipeline
        regions = regions.loc[X.index]

    if save_X_name:
        log_text("Saving Xs for future use...")
        tosave = pd.concat([locations, X], axis=1)
        if regions_cond:
            tosave = pd.concat([tosave, regions], axis=1)
        tosave.to_pickle(save_X_path)
        robust_chmod(save_X_path, 0o774)

    return X, locations, Y, regions


        
