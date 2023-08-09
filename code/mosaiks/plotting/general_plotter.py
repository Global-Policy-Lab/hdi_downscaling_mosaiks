import pickle
from os.path import dirname, isfile, join, exists

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.vectorized import contains

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import matplotlib
import matplotlib.patches as mpatches

import cartopy.crs as ccrs

import seaborn as sns
import sklearn.metrics as metrics
from mosaiks import config
from mosaiks.solve import solve_functions as solve
from mosaiks.solve.interpret_results import interpret_kfold_results
from mosaiks.label_utils.plotting_utils import get_world_map

from mosaiks.utils.io import create_dir

from mosaiks.utils.logging import log_text, robust_chmod

from ..utils import OVERWRITE_EXCEPTION

def _adjust_val_names_str(val_names):
    if isinstance(val_names, str):
        val_names = [val_names]
    return val_names


def _savefig(
    fig, save_dir, save_str=None, app_name=None, val=None, prefix=None, suffix=None,
    tight_layout=False, overwrite=True
):
    if tight_layout:
        fig.tight_layout()
    # This is here for backward-compatibility; we may want to delete 
    # the contents of this if statement in the future
    if save_str is None:
        assert all([app_name, val, prefix, suffix]), \
                "No save_str given and at least one of app_name, val, prefix," + \
                "and suffix is None."
        save_str = join([save_dir, 
                         "{}_{}_{}_{}.png".format(prefix, app_name, val, suffix)])
    else:
        save_str = join(save_dir, "{}_{}.png".format(prefix, save_str))
    if isfile(save_str) and (not overwrite):
        raise OVERWRITE_EXCEPTION
    if not exists(save_dir):
        create_dir(save_dir)
       
    fig.savefig(save_str, dpi=300)
    robust_chmod(save_str)


def _save_fig_data(data, save_dir, save_str, app_name, val, prefix, suffix,
                   overwrite=True):
    if save_str is None:
        assert all([app_name, val, prefix, suffix]), \
                "No save_str given and at least one of app_name, val, prefix," + \
                "and suffix is None."
        data_str = join(save_dir,
                        "data",
                        "{}_{}_{}_{}.data".format(prefix, app_name, val, suffix))
    else:
        data_str = join(save_dir, "data", "{}_{}.data".format(prefix, save_str))
        
    if isfile(data_str) and (not overwrite):
        raise OVERWRITE_EXCEPTION
    
    if not exists(join(save_dir, "data")):
        create_dir(join(save_dir, "data"))

    with open(data_str, "wb") as f:
        pickle.dump(data, f)
        
    robust_chmod(data_str)


def _save_hyperparams_csv(data, save_dir, app_name, val, prefix, suffix, colnames):
    data_str = join(save_dir, "{}_{}_{}_{}".format(prefix, app_name, val, suffix))
    np.savetxt(data_str + ".csv", data, delimiter=",", fmt="%i", header=colnames)


def _get_bounds(bounds, data):
    """Helper func to return data bounds if
    no bounds specified; otherwise return 
    specified bounds."""
    bounds_out = []
    if bounds[0] is None:
        bounds_out.append(data.min())
    else:
        bounds_out.append(bounds[0])
    if bounds[1] is None:
        bounds_out.append(data.max())
    else:
        bounds_out.append(bounds[1])

    return bounds_out


def scatter_preds(
    y_preds,
    y_true,
    appname=None,
    title=None,
    ax=None,
    c=None,
    s=0.08,
    alpha=0.4,
    edgecolors="none",
    bounds=None,
    linewidth=0.75,
    axis_visible=False,
    fontsize=6.3,
    despine=True,
    rasterize=False,
    is_ACS=False,
):
    """ give a scatter plot of predicted vs. actual values, and set the title as
    specified in the arguments + add some info on the metrics in the title.
    y_true is a vector of true values, y_preds the corresponding predictions."""
    if ax == None:
        fig, ax = plt.subplots(figsize=(6.4, 6.4))

    # first pull defaults from app
    if appname is not None:
        pa = config.plotting
        if not is_ACS:
            this_bounds = pa["scatter_bounds"][appname]

    # now override if you specified
    if bounds is not None:
        this_bounds = bounds
    if alpha is not None:
        this_alpha = alpha

    this_bounds = _get_bounds(this_bounds, np.hstack((y_true, y_preds)))
    # scatter and 1:1 line
    ax.scatter(
        y_preds,
        y_true,
        alpha=this_alpha,
        c=c,
        s=s,
        edgecolors=edgecolors,
        rasterized=rasterize,
    )
    ax.plot(this_bounds, this_bounds, color="k", linewidth=linewidth)

    # fix up axes shape
    ax.set_ylim(*this_bounds)
    ax.set_xlim(*this_bounds)
    ax.set_aspect("equal")
    ax.set_title(title)
    if not axis_visible:
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
    if despine:
        sns.despine(ax=ax, left=True, bottom=True)

    # add r2
    ax.text(
        0.05,
        1,
        "$R^2 = {:.2f}$".format(metrics.r2_score(y_true, y_preds)),
        va="top",
        ha="left",
        transform=ax.transAxes,
        fontsize=fontsize,
    )

    return ax


def metrics_vs_size(
    results,
    best_lambdas,
    num_folds,
    val_names,
    num_vector,
    xtitle,
    crits="r2_score",
    app_name=None,
    save_dir=None,
    prefix=None,
    suffix=None,
    figsize=(10, 5),
    overwrite=False,
):

    """Plot metrics (e.g. r2) vs number of training observations used to train model. 

    Args:
     results (list of dictionaries) : e.g. [{'mse': 117.05561471285215, 'r2_score': 0.875037330527241},
         {'mse': 119.84752736626068, 'r2_score': 0.8735189806862442}]
     best_lambdas (1darray-like) : chosen hyperparameters
     num_folds (scalar) : number of folds stored in the results dictionary.
     crit (str or list of str) : Names of criteria that you want to plot (e.g. 'r2_score') for
        each outcome.
     val_names (str or list of str) : Names of outcome(s). If multiple outcomes, this must be
         a list 
     num_vector (list of scalars) : list of scalars to loop over and re-train. E.g. for plotting performance
        against number of training samples, this is a vector of sample sizes.
     xtitle (str) : Either "train set size" or "number of features", depending on which you're plotting on the x axis 
     crits (str or list of str) : Names of criteria that you want to plot (e.g. 'r2_score') for
                  each outcome.
     app_name (str) : The name of the application (e.g. 'housing'). Only needed if saving
     save_dir (str) : Path to directory in which to save output files. If None, no figures will be saved.
     prefix (str) : Filename prefix identifying what is being plotted (e.g. test_outof_cell_r2). Only
         needed if figure is being saved
     suffix (str) : The suffix containing the grid and sample parameters which will be appended to the
         filename when saving, in order to keep track of various sampling and gridding schemes.
     overwrite (bool, optional) : If ``overwrite==False`` and the filename that we will 
         save to exists, it will raise an error
    Returns:
    None
    """
    val_names = _adjust_val_names_str(val_names)

    for j, val in enumerate(val_names):

        # initialize stack of outcomes
        yvals_by_fold = []

        # initialize plot
        fig, ax = plt.subplots(figsize=figsize)

        # loop over each fold, store metric and plot
        for i in range(num_folds):
            yvals = [res[i][j][crits[j]] for res in results]
            yvals_by_fold.append(yvals)

            ax.plot(num_vector[:], yvals[:], label="fold {0}".format(i))
        ax.set_xscale("log")
        ax.set_title("Performance vs. " + xtitle + " " + val)
        ax.set_xlabel(xtitle)
        ax.set_ylabel(crits[j])
        ax.legend()

        if save_dir is not None:
            _savefig(fig, save_dir, app_name, val, prefix, suffix, overwrite=overwrite)

            # save pickle of data
            to_save = {
                "y_vals": yvals_by_fold,
                "x_vals": num_vector,
                "best_lambda": best_lambdas,
            }
            _save_fig_data(
                to_save, save_dir, app_name, val, prefix, suffix, overwrite=overwrite
            )

    return None


def make_kde(x, y, scale=1, gridsize=100):
    """
    Function to manually create a KDE, which can then be plotted using
    plt.contourf. This function is handy for playing around with KDE bandwidth
    and other parameters without have to interface with seaborn's implementation.
    Maybe want to move this function elsewhere given that it is not currently
    called by any other function in this script.
    
    Based on
    https://stackoverflow.com/questions/50917216/log-scales-with-seaborn-kdeplot
    """
    
    from scipy import stats
    
    n = x.shape[0]
    d = 2
    scotts_factor = n**(-1./(d+4))
    
    kde = stats.gaussian_kde([x, y], bw_method=scotts_factor * scale)
    
    xx, yy = np.mgrid[min(x):max(x):(max(x)-min(x))/gridsize,
                      min(y):max(y):(max(y)-min(y))/gridsize]
    
    
    
    density = kde(np.c_[xx.flat, yy.flat].T).reshape(xx.shape)
    
    return xx, yy, density
    
    

def performance_density(
    kfold_results,
    model_info,
    val,
    lims={},
    save_dir=None,
    save_str=None,
    app_name=None,
    suffix=None,
    kind="kde",
    bw="scott",
    alpha=0.25,
    cut=3,
    size=10,
    joint_only=False,
    show_r2=True,
    show_N=True,
    drop_outliers=False,
    drop_0s=False,
    **plotter_kws
):
    """Plots a KDE plot of OOS preds across all folds vs obs.

        Args:
            kfold_results (dict of ndarray) :
                As returned using kfold_solve()
            model_info (str) :
                To append to title of the scatter plot,
                e.g. could pass in formation about which solve...etc it was.
            val (str or list of str):
                An ordered list of names of the outcomes in this model. If not
                multiple outcomes, this can be string. Otherwise must be a list of strings
                of length n_outcomes
            lims (dict of 2-tuple) : Apply lower and upper bounds to KDE plot for a particular val.
                The format of this dict is val : (lower_bound,upper_bound). If no lim is set
                for a particular val, the default is the lower and upper bound of the observed
                and predicted outcomes combined.
            save_dir (str) : Path to directory in which to save output files. If None, no figures will be saved.
            app_name (str) : The name of the application (e.g. 'housing'). Only needed if saving
            suffix (str) : The suffix containing the grid, sample, and featurization parameters
                which will be appended to the filename when saving, in order to keep track of
                various sampling and gridding schemes. Only needed if saving
            kind (str) : Type of plot to draw. Default is KDE. Options:
                { “scatter” | “reg” | “resid” | “kde” | “hex”
            bw (‘scott’ | ‘silverman’ | scalar | pair of scalars, optional) : Bandwidth to use for kernel in kde
                plots. Default is 'scott'. Only implemented for kind='kde'
            cut (numeric) : Kernel is set to go to 0 at min/max data -/+ cut*bw. Only implemented for kind='kde'
            joint_only (boolean). Plot only the scatter/KDE plot (i.e., the joint
                plot) with no marginals distributions?
            show_r2 (boolean). Plot the R2?
            show_N (boolean). Plot the number of observations?
            drop_outliers (boolean). Only plot the 5th through 95th percentile?
            plotter (plotting function). The plotting function to use
            drop_0s (boolean). Drop observatiosn that are 0?
    """
    # code is flexible for looping over lists of results (for regional models)--
    # if a single set of results is passed, it is coerced into a list for
    # compatibility.
    if not isinstance(kfold_results, list):
        kfold_results = [kfold_results]
        
    val = _adjust_val_names_str(val)

    # get metrics and preds for best hyper parameters
    best_preds = np.vstack(
        [interpret_kfold_results(r, crits="r2_score")[2] for r in kfold_results]
    )

    # flatten over fold predictions
    preds = np.vstack([solve.y_to_matrix(i) for i in best_preds.squeeze()])
    
    truth = np.vstack(
        np.hstack([[solve.y_to_matrix(i) 
                    for i in r["y_true_test"].flatten().squeeze()]
         for r in kfold_results])
    )

    # loop over all outcome dimensions
    n_outcomes = preds.shape[1]
    for i in range(n_outcomes):

        this_truth = truth[:, i]
        this_preds = preds[:, i]
        this_val = val[i]
 
        if drop_0s:
            drops = np.concatenate([np.argwhere(this_truth == 0),
                                    np.argwhere(this_preds == 0)])
            this_truth = np.delete(this_truth, drops)
            this_preds = np.delete(this_preds, drops)

        # calc r2 before clipping
        if show_r2:
            r2 = metrics.r2_score(this_truth, this_preds)
            
        if show_N:
            N = this_truth.shape[0]

        # set axis limits for kde plot
        if this_val in lims.keys():
            this_lims = lims[this_val]
        else:
            # select the min and max of input data, expanded by a tiny bit
            offset = (
                max(
                    [
                        this_truth.max() - this_truth.min(),
                        this_preds.max() - this_preds.min(),
                    ]
                )
                / 1000
            )
            
            if drop_outliers:
                this_min = min(
                    [np.percentile(t, 5) for t in [this_preds, this_truth]]
                ) - offset
                
                this_max = max(
                    [np.percentile(t, 95) for t in [this_preds, this_truth]]
                ) - offset
                
            else:
                this_min = min([this_preds.min(), this_truth.min()]) - offset
                this_max = max([this_preds.max(), this_truth.max()]) + offset
                
            this_lims = (this_min, this_max)

        log_text("Plotting {}...".format(this_val))

        # note that below code clips to axes limits before running kernel
        # so if you clip below a large amount of data, that data will be
        # ignored in the plotting (but not in the r2)
        marginal_kws = {}
        if kind == "kde":
            marginal_kws["bw"] = bw
            marginal_kws["clip"] = this_lims
            marginal_kws["cut"] = cut

        # extend the drawing of the joint distribution to the extremes of the
        # data
        joint_kws = marginal_kws.copy()
        if kind == "kde":
            joint_kws["extend"] = "both"
            joint_kws["shade"] = True
            joint_kws["cmap"] = "binary"
            joint_kws["shade_lowest"] = False
        elif kind == "scatter":
            joint_kws["alpha"] = alpha

        with sns.axes_style("white"):
            if joint_only:
                jg = sns.JointGrid(
                    this_preds,
                    this_truth,
                    height=size,
                    xlim=this_lims,
                    ylim=this_lims
                )
                
                if kind == "kde":
                    plotter = sns.kdeplot
                elif kind == "scatter":
                    plotter = sns.scatterplot
                    joint_kws["edgecolor"] = "tab:blue"
                
                jg.plot_joint(plotter, **joint_kws)
                
                # do not plot the marginal distributions
                jg.ax_marg_x.set_axis_off()
                jg.ax_marg_y.set_axis_off()
                
            else:
                jg = sns.jointplot(
                    this_preds,
                    this_truth,
                    kind=kind,
                    height=size,
                    xlim=this_lims,
                    ylim=this_lims,
                    joint_kws=joint_kws,
                    marginal_kws=marginal_kws
                )

        ## add 1:1 line
        jg.ax_joint.plot(this_lims, this_lims, "k-", alpha=0.75)
        jg.ax_joint.set_xlabel("Predicted", fontsize=30)
        jg.ax_joint.set_ylabel("Observed", fontsize=30)
        jg.ax_joint.xaxis.set_tick_params(labelsize=20)
        jg.ax_joint.yaxis.set_tick_params(labelsize=20)

        if show_r2:
            jg.ax_joint.text(
                0.05, 0.95, "r2_score: {:.2f}".format(r2),
                transform=jg.ax_joint.transAxes,
                fontsize=14
            )
            
        if show_N:
            jg.ax_joint.text(
                0.05, 0.90, "N: {:,}".format(N),
                transform=jg.ax_joint.transAxes,
                fontsize=14
            )
        
        ## calc metrics
        plt.suptitle(
            "{} Model OOS Performance w/ k-fold CV ({})".format(
                this_val.title(), model_info.title()
            )
        )
        if save_dir:
            fig = plt.gcf()
            _savefig(
                fig,
                save_dir,
                save_str,
                app_name,
                this_val,
                "predVobs_kde",
                suffix,
                tight_layout=True,
            )
            
            kde_data = {"truth": this_truth, "preds": this_preds}
            _save_fig_data(
                kde_data, save_dir, save_str, app_name, this_val, "predVobs_kde", suffix
            )


def points_to_bin(x, y, vals, scale=.1, coastlines_shp=None, plot_crs=None):
    """bins points over 2d space with bin sizes specified by scale
        args:
            x,y: nx1 arrays of locations in 1 dimension each
            vals: nx1 array of values to be averaged
            scale: the edge of a bin/box in {x,y} units.
            coastlines_shp: GeoPandas DataFrame. Shapefile of the coastlines for the
                region you want to plot.
            plot_crs: NOT CURRENTLY IMPLEMENTED.
                the CRS you want your plot in. This argument has not been
                tested thoroughly and should be used with caution. Your best bet
                is to make sure your coastlines shapefile is in the same CRS as
                the CRS of the plot you are trying to create before feeding the
                shapefile into this function.
        returns:
            x0, y0: kx1, mx1 arrays of the x and y gridpoints
            vals_grid: (m-1)x(k-1) resulting aggregated values
        """
    x_range = np.max(x) - np.min(x)
    y_range = np.max(y) - np.min(y)
    bin_shapes = [int(y_range / scale), int(x_range / scale)]
    
    sums_grid, y0, x0 = np.histogram2d(y, x, bins=bin_shapes, weights=vals)
    counts, y1, x1 = np.histogram2d(y, x, bins=bin_shapes)
    
    # special handling if a regional shapefile is given.
    # This allows to distinguish between places with 0 observations and places
    # that are outside the area we have data for in plotting
    if coastlines_shp is not None:
        log_text( ("Subsetting to coastlines is experimental functionality." +
                      "Nothing guarrantees that your shapefiles will be in " +
                      "consistent CRSes. " +
                      "If you are using this functionality, ask Simon to go " +
                      "implement subsetting the data before passing it to " +
                      "this function. Remind him that he promised Luke he " +
                      "would do this."), print_text="warn")
                      
        assert coastlines_shp.shape[0] == 1, "Expected a single-region shapefile"
        
        # plot_crs = plot_crs.proj4_params
        # coastlines_shp.to_crs("WGS84", inplace=True)
        
        # ensure common CRS
        # if coastlines_shp.crs != plot_crs:
            # Note: this will not work for polygon-aggregated data.
        #     coastlines_shp.to_crs(plot_crs.proj4_params, inplace=True)
            
        # get the list of points that are inside the region being plotted
        x1_grid, y1_grid = np.meshgrid(x1, y1)
        inbounds = contains(coastlines_shp["geometry"][0], x1_grid, y1_grid)
        # inbounds = inbounds.reshape()
        inbounds = inbounds[:counts.shape[0], :counts.shape[1]]

    vals_grid = sums_grid / counts
    
    if coastlines_shp is not None:
        vals_grid = np.where((inbounds) & (counts == 0), 0, vals_grid)    
    
    vals_grid = np.ma.masked_invalid(vals_grid)
    
    return x0, y0, vals_grid  



def make_classification_raster(values,x,y, scale): 
    """
    This function is built off points_to_bin(). For classification, we create a raster grid for each class.
    
    We then use np.argmax to grab the raster layer with the highest mean value of the given classes. This corresponds to the class
    that is most represented at each grid cell.
    
    TODO: implement masking according to a coastlines shapefile as in points_to_bin().
    
    """

    factors = np.unique(values)
    val_grids = []
    for factor in factors:
        is_class = (factor == values).astype(int).flatten()
        _,_, val_grid = points_to_bin(x, y, is_class, scale=scale)
        val_grids.append(val_grid)

    mask = val_grids[0].mask
    arr = np.argmax(val_grids, axis=0) #
    arr = np.ma.masked_array(arr, mask)

    return arr



def set_default_crs(ext, polygon=False):
    """
    Helper function to set a CRS for mapping given the extent of the area to
    be mapped. The logic behind having different defaults for different extents
    is that different CRSs make for more aesthetically appealing/geographically
    accurate plots for different sized areas. You can always override this
    default by passing your preferred crs to the `plot_crs` argument of the
    plotting function.
    Current defaults are Mercator for smaller maps (less than 20x20 degree
    extent) and PlatteCarree for larger maps (greater than or equal to 20x20
    degree extent. We may want to play around with these defaults, as well as
    the map sizes they apply to.
    Args:
        ext (tuple):
            tuple of xmin, xmax, ymin, ymax
        polygon (bool):
            return a Proj4 string (for used in polygon plotting)?
    """
    xmin, xmax, ymin, ymax = ext
    assert xmin < xmax, "extent not well-defined: xmin >= xmax"
    assert ymin < ymax, "extent not well-defined: ymin >= ymax"

    xrange = xmax - xmin
    yrange = ymax - ymin

    
    if xrange > 20 or yrange > 20: # we may want to play around w/ these
        if polygon:
            return "EPSG:4326"
        else:
            return ccrs.PlateCarree()
    else:
        if polygon:
            return "EPSG:3395"
        else:
            return ccrs.Mercator()


def spatial_raster_obs_v_pred(
    kfold_results,
    model_info,
    val,
    res = 0.3,
    residuals=False,
    save_dir=None,
    save_str=None,
    app_name=None,
    suffix=None,
    subplot_stack="vertical",
    figsize=None,
    crit="r2_score",
    cbar_shrink=.8,
    plot_crs=None,
    plot_coastlines=True,
    coastlines_shp=None,
    contour=False,
    **kwargs
):
    """Plots side-by-side spatial rasters of observed and predicted values.
    
    Note that plotting behavior is different when model_info = "OVR_classifier". 

        Args:
            kfold_results (dict of ndarray) :
                As returned using kfold_solve()
            latlons (nx2 2darray) : lats (first col), lons (second col)
            model_info (str) :
                To append to title of the scatter plot,
                e.g. could pass in formation about which solve...etc it was.
            val (str or list of str):
                An ordered list of names of the outcomes in this model. If not
                multiple outcomes, this can be string. Otherwise must be a list of strings
                of length n_outcomes
            res (float):
                The resolution to plot the raster at. In units of degrees.
            residuals (boolean):
                If True, produce a single plot showing the prediction residuals.
                If False, produce side-by-side plots showing predicted and
                observed values.
            save_dir (str) : Path to directory in which to save output files. If None, no figures will be saved.
            app_name (str) : The name of the application (e.g. 'housing'). Only needed if saving
            suffix (str) : The suffix containing the grid, sample, and featurization parameters
                which will be appended to the filename when saving, in order to keep track of
                various sampling and gridding schemes. Only needed if saving
            cbar_shrink (float): scalar to adjust the size of the colorbar on the plots
            plot_crs (Cartopy CRS object): Use to output plots in a particular CRS. If None, calls `set_default_crs()` to choose a CRS depending on extent of map being plotted (Mercator for small maps, PlatteCarree for large maps).
            plot_coastlines (bool): Plot the outlines of the coasts on the plot?
            coastlines_shp (GeoPandas DataFrame): Geopandas DataFrame containing
                spatial information for the coastlines you want to draw. If 
                `plot_coastlines` is set to `True` and no value is specified,
                coastlines will be drawn using `ax.coastlines()` from Cartopy.
            contour (bool): use `plt.contourf()` instead of `plt.imshow()` to plot
                the raster? The contour plot will plot faster for higher-resolution
                rasters, but is slow for large spatial extents.
    """
    # code is flexible for looping over lists of results (for regional models)--
    # if a single set of results is passed, it is coerced into a list for
    # compatibility.
    if not isinstance(kfold_results, list):
        kfold_results = [kfold_results]    
        
    subplot_stack_dict = {"vertical" : (2,1), "horizontal": (1, 2) }
    coastlines_resolution="50m"
    val = _adjust_val_names_str(val)
    
    
    # get metrics and preds for best hyper parameters
    best_preds = np.vstack(
        [interpret_kfold_results(r, crits=crit)[2] for r in kfold_results]
    )

    # flatten over fold predictions
    preds = np.vstack([solve.y_to_matrix(i) for i in best_preds.squeeze()])


    truth = np.vstack(
        np.hstack([[solve.y_to_matrix(i) 
                    for i in r["y_true_test"].flatten().squeeze()]
         for r in kfold_results])
    )

    # get latlons in same shuffled, cross-validated order from kfold results
    ll = np.vstack(
        [np.vstack(r["locations_test"]) 
         for r in kfold_results])

    x = ll[:,0]
    y = ll[:,1]

    # set up projection and CRS for maps
    ext = (x.min(), x.max(), y.min(), y.max())

    if plot_crs is None:
        plot_crs = set_default_crs(ext)
    src_crs = ccrs.PlateCarree()
    
    if model_info != "OVR_classifier":
        resid = truth - preds
        vmin = kwargs.pop("vmin", np.percentile(truth, 1, axis=0))
        vmax = kwargs.pop("vmin", np.percentile(truth, 99, axis=0))
        # for residual plot, make bounds symmetrical, so 0 is at the middle of 
        # the color bar.
        bound = max(abs(vmin), abs(vmax))

        for vx, v in enumerate(val):
            if residuals:
                # plot residuals
                fig, ax = plt.subplots(figsize=figsize,
                                       constrained_layout=True,
                                       subplot_kw={'projection':plot_crs}
                                      )

                resid_val = resid[:, vx]

                resid_x, resid_y, resid_grid = points_to_bin(x, y, resid_val,
                                                             res, coastlines_shp,
                                                             plot_crs)

                if kwargs.get("cmap"):
                    cmap = kwargs.pop("cmap")
                else:
                    cmap = "PuOr" #use default cmap
                    
                if contour:
                    # set the bounds for the filled contour levels.
                    # this could be turned into kwargs if desired.
                    maxpt = 99
                    minpt = 0
                    # set the step size between contour levels (make this very
                    # small so it looks nearly continuous). This could also be
                    # a kwarg.
                    step=0.005
                    
                    maxval = np.percentile(resid_grid.compressed(), maxpt)
                    minval = np.percentile(resid_grid.compressed(), minpt)
                    
                    # re-set maxval and minval so that the colorbar is set at 0
                    maxval = max(maxval, abs(minval))
                    minval = min(-maxval, minval)
                    
                    contour_lvls = np.arange(minval, maxval, step=step)
                    
                    r = ax.contourf(
                        resid_x[:resid_grid.shape[1]],
                        resid_y[:resid_grid.shape[0]],
                        resid_grid,
                        levels=contour_lvls,
                        cmap=cmap)
                
                else:
                    r = ax.imshow(resid_grid,
                                  origin='lower',
                                  vmin=-bound,
                                  vmax=bound,
                                  cmap=cmap,
                                transform=src_crs,
                                  extent=ext,
                                  **kwargs)
                    
                    ax.grid(None)
                    ax.set_extent(ext)

                cbar = fig.colorbar(r, shrink=cbar_shrink)

                # Option to add coastlines
                if plot_coastlines:
                    ax.coastlines(resolution=coastlines_resolution, color='black', linewidth=0.1)

                ax.set_title("Prediction residuals")

            else:
                # plot obs and preds
                fig, ax = plt.subplots(
                                       subplot_stack_dict[subplot_stack][0],
                                       subplot_stack_dict[subplot_stack][1],
                                       figsize=figsize, 
                                       constrained_layout=True,
                                      subplot_kw={'projection':plot_crs})

                truth_val = truth[:, vx]
                pred_val = preds[:, vx]
                
                truth_x, truth_y, truth_grid = points_to_bin(x, y, truth_val, res,
                                                 coastlines_shp, plot_crs)
                
                pred_x, pred_y, pred_grid = points_to_bin(x, y, pred_val,res,
                                                coastlines_shp, plot_crs)
                
                if contour:
                    maxpt = 99
                    minpt = 0
                    maxval = max(
                        np.percentile(truth_grid.compressed(), maxpt),
                        np.percentile(pred_grid.compressed(), maxpt)
                    )
                    minval = min(
                        np.percentile(truth_grid.compressed(), minpt),
                        np.percentile(pred_grid.compressed(), minpt)
                    )
                    # set the step size between contour levels (make this very
                    # small so it looks nearly continuous). This could also be
                    # a kwarg.
                    step=0.005
                    contour_lvls = np.arange(minval, maxval, step=step)
                    
                    if kwargs.get("cmap"):
                        cmap = kwargs.pop("cmap")
                    else:
                        cmap = "viridis" #use default cmap
                    
                    r0 = ax[0].contourf(
                        truth_x[:truth_grid.shape[1]],
                        truth_y[:truth_grid.shape[0]],
                        truth_grid,
                        levels=contour_lvls,
                        cmap=cmap
                    )
                    r1 = ax[1].contourf(
                        pred_x[:pred_grid.shape[1]],
                        pred_y[:pred_grid.shape[0]],
                        pred_grid,
                        levels=contour_lvls,
                        cmap=cmap
                    )
                
                else:
                    r0 = ax[0].imshow(truth_grid,
                                      origin= 'lower',
                                      vmin=vmin[vx],
                                      vmax=vmax[vx],
                                      extent = ext,
                                      transform = src_crs,
                                      interpolation = "none",
                                      **kwargs)

                    r1 = ax[1].imshow(pred_grid,
                                      origin='lower',
                                      vmin=vmin[vx],
                                      vmax=vmax[vx],
                                      extent = ext,
                                      transform = src_crs,
                                      interpolation = "none",
                                      **kwargs)
                    
                    ax[0].set_extent(ext)
                    ax[1].set_extent(ext)

                
                cbar = fig.colorbar(r0, ax=ax.ravel().tolist(), shrink=cbar_shrink)



                ax[0].set_title("Observed")
                ax[1].set_title("Predicted")

                if plot_coastlines:
                    ax[0].coastlines(resolution=coastlines_resolution, color='black', linewidth=0.1)
                    ax[1].coastlines(resolution=coastlines_resolution, color='black', linewidth=0.1)


    else: ## Here we make customized raster plots with results from discrete classification
        for vx, v in enumerate(val):

            truth_val = truth[:, vx]
            pred_val = preds[:, vx]

            truth_grid = make_classification_raster(truth, x, y, scale=res)
            pred_grid = make_classification_raster(preds, x, y, scale=res)
            resid_grid = (truth_grid == pred_grid).astype(int)

            if residuals:
                fig, ax = plt.subplots(figsize=figsize,
                                       constrained_layout=True,
                                     subplot_kw = {'projection':plot_crs}
                                      )

                if kwargs.get("cmap"):
                    cmap = kwargs.pop("cmap")
                else:
                    cmap = "RdYlGn" #use default cmap


                r = ax.imshow(
                    resid_grid,
                    origin = "lower",
                    cmap = "RdYlGn", # not colorblind friendly
                    extent = ext,
                    transform = src_crs,
                    interpolation = "none",
                    **kwargs
                )

                ax.set_extent(ext)

                # In these lines we create a custom legend, since imshow wants to use a colorbar we manually code this
                factors = ["incorrect", "correct"]
                norm = matplotlib.colors.Normalize(vmin=0, vmax=len(factors)-1)
                cmap = matplotlib.cm.get_cmap(cmap)
                patches = [mpatches.Patch(color=cmap(norm(i)), label=factor) for i, factor in enumerate(factors)]
                ax.set_title("Prediction errors")
                ax.legend(handles=patches, fontsize="large", bbox_to_anchor=(1.2, 1))


            else:
                fig, ax = plt.subplots(
                                       subplot_stack_dict[subplot_stack][0],
                                       subplot_stack_dict[subplot_stack][1],
                                       figsize=figsize,
                                       constrained_layout=True,
                                       subplot_kw={'projection':plot_crs})

                if kwargs.get("cmap"):
                    cmap = kwargs.pop("cmap")
                else:
                    cmap = "Dark2" #use default cmap

                factors = np.unique(truth_val)
                min_class = 0 
                max_class = len(factors)-1
                norm = matplotlib.colors.Normalize(vmin= min_class, vmax= max_class)                

                r0 = ax[0].imshow(
                    truth_grid,
                    origin="lower",
                    cmap=cmap,
                    extent = ext,
                    transform = src_crs,
                    interpolation = "none",
                    vmin= min_class, 
                    vmax= max_class,
                    **kwargs
                )

                r1 = ax[1].imshow(
                    pred_grid,
                    origin="lower",
                    cmap=cmap,
                    extent = ext,
                    transform = src_crs,
                    interpolation = "none",
                    vmin= min_class, 
                    vmax= max_class,
                    **kwargs
                )

                # In these lines we create a custom legend, since imshow wants to use a colorbar we manually code this
                cmap = matplotlib.cm.get_cmap(cmap)
                patches = [mpatches.Patch(color=cmap(norm(i)), label=factor) for i, factor in enumerate(factors)]
                ax[0].set_title("Observed")
                ax[1].set_title("Predicted")
                ax[0].set_extent(ext)
                ax[1].set_extent(ext)

                ax[1].legend(handles=patches, fontsize="large", bbox_to_anchor=(1.2, 1))


    fig.suptitle(v.title())

    if save_dir:
        data = {
            "lon": ll[:, 0],
            "lat": ll[:, 1],
            "truth": truth[:, vx],
            "preds": preds[:, vx],
        }
        if residuals:
            _savefig(fig, save_dir, save_str, app_name, v,
                     "outcomes_raster_resid", suffix)
            _save_fig_data(data, save_dir, save_str, app_name, v,
                           "outcomes_raster_resid", suffix)
        else:
            _savefig(fig, save_dir, save_str, app_name, v,
                     "outcomes_raster_obsAndPred", suffix)
            _save_fig_data(data, save_dir, save_str, app_name, v,
                           "outcomes_raster_obsAndPred", suffix)


def spatial_polygon_obs_v_pred(
    kfold_results,
    model_info,
    val,
    c,
    c_app,
    residuals=False,
    save_dir=None,
    save_str=None,
    app_name=None,
    suffix=None,
    figsize=None,
    crit="r2_score",
    cbar_shrink = .8,
    subplot_stack = "vertical",
    plot_crs = None,
    plot_bounds = None,
    **kwargs
):

    """Plots side-by-side spatial polygons of observed and predicted values.

        Args:
            kfold_results (dict of ndarray) :
                As returned using kfold_solve()
            polygon_ids (nx1 array) :
                Polygon unique identifiers. Will be merged with shapefile
            model_info (str) :
                To append to title of the scatter plot,
                e.g. could pass in formation about which solve...etc it was.
            val (str or list of str):
                An ordered list of names of the outcomes in this model. If not
                multiple outcomes, this can be string. Otherwise must be a list of strings
                of length n_outcomes
            residuals (boolean):
                If True, produce a single plot showing the prediction residuals.
                If False, produce side-by-side plots showing predicted and
                observed values.
            save_dir (str) : Path to directory in which to save output files. If None, no figures will be saved.
            app_name (str) : The name of the application (e.g. 'housing'). Only needed if saving
            suffix (str) : The suffix containing the grid, sample, and featurization parameters
                which will be appended to the filename when saving, in order to keep track of
                various sampling and gridding schemes. Only needed if saving
            plot_bounds (str): ("world_map" or None. Default None.) "world_map" as bounds for xlim/ylim as default. Otherwise set to data extent
    """
    
    if c_app.get("shp_file_name") == None:
        log_text("No shp file given in c_app, no map will be plotted")
        
        return
    # code is flexible for looping over lists of results (for regional models)--
    # if a single set of results is passed, it is coerced into a list for
    # compatibility.
    if not isinstance(kfold_results, list):
        kfold_results = [kfold_results]
    
        
    subplot_stack_dict = {"vertical" : (2,1), "horizontal": (1, 2) }

    val = _adjust_val_names_str(val)

    # get metrics and preds for best hyper parameters
    best_preds = np.vstack(
        [interpret_kfold_results(r, crits=crit)[2] for r in kfold_results]
    )

    # flatten over fold predictions
    preds = np.vstack([solve.y_to_matrix(i) for i in best_preds.squeeze()])

    truth = np.concatenate([np.concatenate(r['y_true_test']) for r in kfold_results])

    resid = truth - preds

    # get locations in same shuffled, cross-validated order
    ll = np.concatenate([np.concatenate(r['locations_test']) for r in kfold_results]).squeeze()
    

    polygon_id_colname = c_app["polygon_id_colname"]

    results_df = pd.DataFrame({
        polygon_id_colname: ll, 'truth': truth.squeeze(), 'preds': preds.squeeze()
    })
   
    shp_path = join(c.data_dir,
            "int",
            "applications" ,
            c_app["labels_directory"] ,
           c_app["shp_file_name"])
    
    if shp_path.endswith(".p") or shp_path.endswith(".pkl"):
        gpdf = pd.read_pickle(shp_path)
    else:
        gpdf = gpd.read_file(shp_path)
    
    if gpdf.crs is None:
        gpdf.crs = 'epsg:4326'

    try:
        # Try to automatically process a (possibly messy) gpdf
        # This code was originally written when trying to parse the cropped 
        # area shapefile, and seems to cause problems for other shapefiles.
        # This should be removed eventually; leaving it for now while we figure
        # out what other shapefiles do and don't cause problems.        
        ordered_train_gpdf = gpdf.loc[gpdf[polygon_id_colname].isin(ll)]
        assert ordered_train_gpdf.shape[0] != 0, "polygon ids from shapefile don't match inputs... trying to convert shapefile identifiers to integers."

    except: # This is annoying but it deals with the fact that a leading zero was dropped when reading the polygon unique id labels from csv
        try:
            gpdf[polygon_id_colname] = gpdf[polygon_id_colname].astype(int)
        except:
            gpdf[polygon_id_colname] = gpdf[polygon_id_colname].astype("category")
        
        ordered_train_gpdf = gpdf.set_index(polygon_id_colname).loc[ll]


    xmin, ymin, xmax, ymax = ordered_train_gpdf.total_bounds
    ext = (xmin, xmax, ymin, ymax)

    if plot_crs is None:
        plot_crs = set_default_crs(ext, polygon=True)

    if plot_crs != ordered_train_gpdf.crs.to_string():
        ordered_train_gpdf = ordered_train_gpdf.to_crs(plot_crs)

    xmin, ymin, xmax, ymax = ordered_train_gpdf.total_bounds #this now changes with the transformation

    ordered_train_gpdf = ordered_train_gpdf.merge(results_df, on=polygon_id_colname)
    ordered_train_gpdf["resid"] = ordered_train_gpdf["truth"] - ordered_train_gpdf["preds"]

    world_map = get_world_map()
    world_map.crs = 'epsg:4326'
    world_map = world_map.to_crs(plot_crs)
    
    
    # Set plot bounds based on user set param
    if plot_bounds=="world_map":
        xmin, ymin, xmax, ymax = world_map.total_bounds
        log_text(ymin + ", " + ymax)
    else:
        xmin, ymin, xmax, ymax = ordered_train_gpdf.total_bounds #this now changes with the transformation

    vmin = kwargs.pop("vmin", np.percentile(truth, 1, axis=0))
    vmax = kwargs.pop("vmin", np.percentile(truth, 99, axis=0))
    # for residual plot, make bounds symmetrical, so 0 is at the middle of 
    # the color bar.
    bound = max(abs(vmin), abs(vmax))

    # plot obs and preds
    for vx, v in enumerate(val):
        if residuals:
            fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
            
            background = world_map.plot(
                ax = ax, 
                color="gray", 
                alpha = 0.3)

            p = ordered_train_gpdf.plot(
                column = "resid",
                ax=ax,
                vmin=-bound,
                vmax=bound,
                legend=True,
                linewidth=0,
                cmap="PuOr",
                legend_kwds={'shrink': cbar_shrink})
            
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
          
            ax.grid(None)
            ax.set_title("Prediction Residuals")

            ax.axes.get_yaxis().set_visible(False)
            ax.axes.get_xaxis().set_visible(False)

        else:
            fig, ax = plt.subplots(
                subplot_stack_dict[subplot_stack][0],
                subplot_stack_dict[subplot_stack][1],
                figsize=figsize, constrained_layout=True)
            
            background0 = world_map.plot(
                ax = ax[0], 
                color="gray", 
                alpha = 0.3)


            p0 = ordered_train_gpdf.plot(
                column = "truth",
                ax=ax[0],
                vmin = vmin,
                vmax=vmax,
                linewidth = .0,
            )
            
            background = world_map.plot(
                ax = ax[1], 
                color="gray", 
                alpha = 0.3)
    

            p1 = ordered_train_gpdf.plot(
                column = "preds",
                ax=ax[1],
                vmin = vmin,
                vmax=vmax,
                linewidth = 0,
            )
        
            # Normalizer
            norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

            # creating ScalarMappable
            sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
            sm.set_array([])

            fig.colorbar(sm, ax=ax.ravel().tolist(),aspect=25)

            
            ax[0].set_xlim(xmin, xmax)
            ax[0].set_ylim(ymin, ymax)
            ax[1].set_xlim(xmin, xmax)
            ax[1].set_ylim(ymin, ymax)

            ax[0].grid(None)
            ax[1].grid(None)

            ax[0].set_title("Observed")
            ax[1].set_title("Predicted")

            ax[0].axes.get_yaxis().set_visible(False)
            ax[0].axes.get_xaxis().set_visible(False)
            ax[1].axes.get_yaxis().set_visible(False)
            ax[1].axes.get_xaxis().set_visible(False)

        fig.suptitle(v.title())

        if save_dir:
            data = (
                ordered_train_gpdf[[polygon_id_colname, 'truth', 'preds',
                                    'resid']]
            )
            
            if residuals:
                _savefig(fig, save_dir, save_str, app_name, v,
                         "outcomes_polygon_resid", suffix)
                _save_fig_data(data, save_dir, save_str, app_name, v,
                               "outcomes_polygon_resid", suffix)
            else:
                _savefig(fig, save_dir, save_str, app_name, v,
                         "outcomes_polygon_obsAndPred", suffix)
                _save_fig_data(data, save_dir, save_str, app_name, v,
                               "outcomes_polygon_obsAndPred", suffix)



def plot_diagnostics(kfold_results, outcome_name, app, polygon_id_colname, grid,
                     c, c_app, save=True,
                     joint_only=False, show_r2=True, res=0.3):
    """
    Produce scatterplot and maps.
    """
    model = c_app.get("solve_function", "Ridge")
    
    if save:
        save_dir = c.plot_savedir
        create_dir(save_dir)
        save_str = c.fname
    else:
        save_dir = None
        save_str = None
    
    if save_dir is not None or save_str is not None:
        log_text(save_dir)
        
    # scatterplot
    if model != "OVR_classifier":
        performance_density(
            kfold_results,
            model,
            val=outcome_name,
            save_dir=save_dir,
            save_str=save_str,
            app_name=app,
            kind="scatter",
            suffix = grid,
            joint_only=joint_only,
            show_r2=show_r2
        )
    else:
        log_text("No scatter plot for classification solve")

        if type(kfold_results)==dict:
            kfold_results = [kfold_results]
        else:
            log_text("Printing accuracy result for each region...")
        
        for results in kfold_results:
            best_lambda_idx, _, _ = interpret_kfold_results(
                results, "r2_score")
            accuracy = np.mean([fold[0][best_lambda_idx[0][0]]["r2_score"] for fold in results["metrics_test"] ]).round(3)
            log_text(f"Accuracy: {accuracy}") # Note that accuracy is called r2 in the kfold results file but this is incorrect

    # predicted vs. observed plots and residual plots
    if polygon_id_colname: # if polygon level aggregation, try to use shapefiles
                           # to make plots
        if c_app.get("shp_file_name"):
            # obs v pred
            spatial_polygon_obs_v_pred(
                kfold_results,
                c=c,
                c_app=c_app,
                model_info=model,
                val=outcome_name,
                save_dir=save_dir,
                save_str=save_str,
                app_name=app,
                suffix=grid)

            # residuals
            spatial_polygon_obs_v_pred(
                kfold_results,
                c=c,
                c_app=c_app,
                model_info=model,
                val=outcome_name,
                save_dir=save_dir,
                save_str=save_str,
                app_name=app,
                suffix=grid,
                residuals=True)
        else:
            log_text("No shapefile path given in config. Can't make polygon level plot")

    else: #If no polygons given, assume tile level observations and make raster comparison plots
        # obvs v pred
        spatial_raster_obs_v_pred(
            kfold_results,
            model_info=c_app.get("solve_function", "Ridge"),
            val=outcome_name,
            save_dir=save_dir,
            save_str=save_str,
            app_name=app,
            suffix = grid,
            res=res
        )

        # residuals
        spatial_raster_obs_v_pred(
            kfold_results,
            model_info=c_app.get("solve_function", "Ridge"),
            val=outcome_name,
            save_dir=save_dir,
            save_str=save_str,
            app_name=app,
            suffix = grid,
            residuals=True,
            res=res
        )