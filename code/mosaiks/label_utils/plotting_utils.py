import matplotlib as mpl
from matplotlib import pyplot as plt
import pandas as pd
import geopandas as gpd
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
import os
import json
#from skimage import io
import rasterio
from rasterio import warp
from rasterio.plot import plotting_extent
from shapely.geometry import Point


# mpl.rcParams['pdf.fonttype'] = 42


def get_world_map(path_to_shapefile='/shares/maps100/data/raw/applications/plotting_shp_files/ne_10m_land/ne_10m_land.shp'):
    return gpd.read_file(path_to_shapefile)


def plot_label_map_hist(df, label_col, label_name, lat_col = 'lat', lon_col = 'lon', xlims = ([-180,180]), ylims=([-60,74]), s=3, fig_width=20, fig_height=10, width_ratio=1.75, single_row=True):
    
    figsize = (fig_width, fig_height)
    
    fig = plt.figure(figsize=figsize)
    
    # relative figure sizes
    
    gs = fig.add_gridspec(nrows=1, ncols = 2, width_ratios = [width_ratio,1], wspace = 0.2)

    # get map of the world
    
    world_map = get_world_map()
    
    # make the map and plot the label values as points 
    
    ax_map = fig.add_subplot(gs[0,0])
    
    # create an axes on the right side of ax. The width of cax will be 3% of ax and the padding between cax and ax will be fixed at 0.005 inch.
    
    divider = make_axes_locatable(ax_map)
    cax = divider.append_axes("right", size="3%", pad=0.05)
    
    world_map.boundary.plot(ax=ax_map, color = 'gray', alpha = 0.3)
    
    im = ax_map.scatter(df[lon_col],df[lat_col], c = df[label_col], s = s)
    fig.colorbar(im, label = label_name, cax = cax)
    
    ax_map.set_xlim(xlims)
    ax_map.set_ylim(ylims)
    ax_map.set_title(label_name)
    
    # make the histogram of frequency of label values 
    
    ax_hist = fig.add_subplot(gs[0,1])
    sns.distplot(df[label_col], kde=False, ax = ax_hist)
    ax_hist.set_ylabel('Number of tiles')
    ax_hist.set_xlabel(label_name)


def label_df_to_raster_visualization(df, label_name, lon_col = "lon", lat_col = "lat", grid_delta = .1):
    """
    Takes a label dataframe as an input, with lat, lon columns and plots a raster version of the label.
    """
    global_stack = df[[lon_col,lat_col, label_name]].to_numpy()
    step = grid_delta
    min_lon, max_lon = np.min(df[lon_col]), np.max(df[lon_col])
    min_lat, max_lat = np.min(df[lat_col]), np.max(df[lat_col])
    lon_range, lat_range = np.round(np.arange(min_lon, max_lon+step,step),3),  np.round(np.arange(min_lat, max_lat+step,step),3)
    raster = np.full(( len(lat_range), len(lon_range) ), -np.inf)

    for i in range(len(global_stack)):
        lon = global_stack[i,0]
        lat = global_stack[i,1]
        val = global_stack[i,2]

        k = np.where(lat_range == lat)[0][0]
        j = np.where(lon_range == lon)[0][0]

        raster[len(lat_range)-k-1,j] = val
        
    fig, ax = plt.subplots(1, figsize = (20, 20) )
    a = ax.imshow(raster, aspect=raster.shape[1] * 1 /raster.shape[0], cmap="viridis")
    fig.colorbar(a)
    ax.grid(None)


    
def bipolar_log_scale(x, buffer=1):
    """This function allows for the plotting of a log scale in both the negative and positive directions.
    This should be used for visualization purposes only when you have mean 0 data"""
    if x >= 0:
        return np.log(x+buffer)
    if x < 0:
        return - np.log(- (x-buffer))
    
    
### PLANET plotting utls below ####
#Note that these functions require that the planet API is installed and intialized

def planet_raster_to_lat_lon_centroids(raster):
    """
    Takes an input rasterio raster file from Planet and outputs the centroid in epsg 4326 coordinates.
    """
    warp.transform_bounds(raster.crs,{"init":"epsg:4326"}, *raster.bounds)
    
    left, bottom, right, top = warp.transform_bounds(raster.crs,{"init":"epsg:4326"}, *raster.bounds)

    lon = (left+right)/2
    lat = (top+bottom)/2
    return round(lat,4), round(lon,4)

def get_planet_tile_path(src_lat, src_lon):
    
    """
    This function takes a set of epsg 4326 lat, lon coordinates and returns the appropriate raster filepath on Tabei.
    
    This function requires the Planet API to be installed and intiliazed. 
    You will need to run `planet init` at the command line before this function will work.
    
    This function takes a set of epsg 4326 lat, lon coordinates and returns the appropriate raster filepath on Tabei. 
    
    This function is designed to ONLY work on the 2019, Quarter 3 set of planet images.
    
    """
    
    #First we read in the appropriate grid centroid
    directory = "/shares/quarter_3/"
    lookup_df = pd.read_csv("/shares/maps100/data/image_metadata/image_metadata_dense.csv")
    
    one_row = lookup_df[(lookup_df.ymin <= src_lat) & (lookup_df.ymax >= src_lat) & (lookup_df.xmin <= src_lon) & (lookup_df.xmax >= src_lon)]
    
    if len(one_row) > 1:
        print("multiple tile paths found. Check inputs")
        return
    elif len(one_row) == 0:
        print("No planet tile path found with input coords")
        return
    else:
        tile_name = one_row.image_file.iloc[0]
        return directory+tile_name

def plot_planet_tile(lat, lon, crop_to_input_tile = False, show_input_tile=True, show_input_point = True,write_path=None):
    """
    This function takes an input lat and lon and displays the planet tile image associated with the coordinate points.
    
    There are optional parameters to crop the planet tile to the grid tile, show the input tile overlayed on the planet tile, 
    and to show and mark the input point. 
    
    Note that if show_input_tile = True and crop_to_input_tile = True, then the entire plot will be overlayed by a transparent shape.
    """
    point_df = gpd.GeoDataFrame({"geometry" : [Point(lon, lat)]}, crs ="EPSG:4326")
    point_df_dst_crs = point_df.to_crs("EPSG:3857")
    point = point_df_dst_crs.loc[0, "geometry"]
    
    
    
    path = get_planet_tile_path(lat, lon)

    im = io.imread(path)
    raster = rasterio.open(path)
    extent = plotting_extent(raster)
    
    
    fig, ax = plt.subplots(1, figsize=(15,30))

    ax.imshow(im, extent=extent)  

    if crop_to_input_tile:
        
        point_df = gpd.GeoDataFrame({"geometry" : [Point(lon, lat)]}, crs ="EPSG:4326")
        src_point = point_df.loc[0, "geometry"]
        src_point_array = np.array([src_point.x, src_point.y])
        centroid_array = np.round((src_point_array - .005), 2) + .005

        tile_df = gpd.GeoDataFrame({"geometry" : [Point(centroid_array).buffer(.005, cap_style=3)]}, crs ="EPSG:4326")
        crop_bounds = tile_df.to_crs("EPSG:3857").total_bounds
        
        ax.set_xlim(crop_bounds[0], crop_bounds[2])
        ax.set_ylim(crop_bounds[1], crop_bounds[3])
    
    else:
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        
    if show_input_point:
        ax.scatter(point.x, point.y, label = "input coordinate point", c="r", s=100, marker = "x", zorder = 2)
        ax.legend()
        
    if show_input_tile:
        src_point = point_df.loc[0, "geometry"]
        src_point_array = np.array([src_point.x, src_point.y])
        centroid_array = np.round((src_point_array - .005), 2) + .005
        
        tile_df = gpd.GeoDataFrame({"geometry" : [Point(centroid_array).buffer(.005, cap_style=3)]}, crs ="EPSG:4326")
        tile_df.to_crs("EPSG:3857").plot(color="orange", alpha=.3, label="input tile area", ax =ax)
    
    if crop_to_input_tile & show_input_tile:
        print("CUSTOM WARNING: \n You have set the function to crop to the input tile and to show the input tile overlayed on the plot. \n Your plot will be entirely covered by a partially transparent rectangle.")


    ax.imshow(im, extent=extent)
    
    ax.axis("off")
    ax.axis("off")
    
    if write_path:
        fig.savefig(write_path, dpi=300)
    
    

def plot_planet_tile_with_gpdFile(lat, lon, gpdFile, label_name, crop_to_input_tile = False, show_input_tile=True, show_input_point = True,):
    """
    
    This function takes an input lat and lon and displays the planet tile image associated with the coordinate points. 
    
    It also takes in a GeoPandas DataFrame (gpdFile) file that is plotted over the raster image. The gpdFile must have shape coordinates in EPSG: 4326".
    
    Finally it requires a label_name (string) that will appear in the legend
    
    It returns a two panel plot. The first panel has the plot with the shapefile and the second panel is the clean image.
    
    There are optional parameters to crop the planet tile to the grid tile, show the input tile overlayed on the planet tile, 
    and to show and mark the input point. 
    
    Note that if show_input_tile = True and crop_to_input_tile = True, then the entire plot will be overlayed by a transparent shape.
    """
    point_df = gpd.GeoDataFrame({"geometry" : [Point(lon, lat)]}, crs ="EPSG:4326")
    point_df_dst_crs = point_df.to_crs("EPSG:3857")
    point = point_df_dst_crs.loc[0, "geometry"]
    
    
    
    path = get_planet_tile_path(lat, lon)

    im = io.imread(path)
    raster = rasterio.open(path)
    extent = plotting_extent(raster)
    
#     gpdf = gpd.GeoDataFrame(gpdFile, crs="EPSG:4326").to_crs(epsg='3857')
    gpdf = gpd.GeoDataFrame(gpdFile, crs="EPSG:4326").to_crs(raster.crs)
    
    fig, ax = plt.subplots(2,1, figsize=(15,30))

    ax[0].imshow(im, extent=extent)
    

    gpdf.plot(ax=ax[0], label= label_name, zorder=1)
    
    
    if crop_to_input_tile:
        point_df = gpd.GeoDataFrame({"geometry" : [Point(lon, lat)]}, crs ="EPSG:4326")
        src_point = point_df.loc[0, "geometry"]
        src_point_array = np.array([src_point.x, src_point.y])
        centroid_array = np.round((src_point_array - .005), 2) + .005

        tile_df = gpd.GeoDataFrame({"geometry" : [Point(centroid_array).buffer(.005, cap_style=3)]}, crs ="EPSG:4326")
        crop_bounds = tile_df.to_crs("EPSG:3857").total_bounds
        
        ax[0].set_xlim(crop_bounds[0], crop_bounds[2])
        ax[0].set_ylim(crop_bounds[1], crop_bounds[3])
        
        ax[1].set_xlim(crop_bounds[0], crop_bounds[2])
        ax[1].set_ylim(crop_bounds[1], crop_bounds[3])
    
    else:
        ax[0].set_xlim(extent[0], extent[1])
        ax[0].set_ylim(extent[2], extent[3])
        
    if show_input_point:
        ax[0].scatter(point.x, point.y, label = "input coordinate point", c="r", s=100, marker = "x", zorder = 2)
        
    if show_input_tile:
        src_point = point_df.loc[0, "geometry"]
        src_point_array = np.array([src_point.x, src_point.y])
        centroid_array = np.round((src_point_array - .005), 2) + .005
        
        tile_df = gpd.GeoDataFrame({"geometry" : [Point(centroid_array).buffer(.005, cap_style=3)]}, crs ="EPSG:4326")
        tile_df.to_crs("EPSG:3857").plot(color="orange", alpha=.3, label="input tile area", ax =ax[0])
        
      
    
    if crop_to_input_tile & show_input_tile:
        print("CUSTOM WARNING: \n You have set the function to crop to the input tile and to show the input tile overlayed on the plot. \n Your plot will be entirely covered by a partially transparent rectangle.")


    ax[0].legend()

    ax[1].imshow(im, extent=extent)
    
    ax[0].axis("off")
    ax[1].axis("off")
    
    
def plot_6_planet_quantile_images(df,col_name, random_seed=123, quantile_bins = [.2,.4,.6,.8,.99], show_axes=False, save_path=None):
    """
    This function can be used to display a set of 6 Planet images that are randomly pulled from a label distribution.
    The quantile bins can be customized so as to pull one image between each percentile. For example, with the default bins,
    we get one image from percentile < 20%, one image from 20% < percentile < 40%, one image from 40% < percentile < 60%, one image from 60% < percentile < 80%,
    and one image percentile > 99%.
    
    For replicability, random seed inputs can be specified.
    
    By default, coordinate axes are not shown with the images. However, this parameter can be set to true. Note that they will be in the raster coordinate system.
    
    Currently, this function is built to show 6 images. Inputting a list or array for "quantile_bins" that is more or less than length 5 will break this function.
    """
    if len(quantile_bins) !=5:
        print("This function can only be used the 5 quantile bins, producing 6 images. Ensure that len(quantile_bins) = 5 for your input parameter.")
        raise Exception
    

    sample_tile_1 = df[df[col_name] <= df[col_name].quantile(quantile_bins[0])].sample(1)
    sample_tile_2 = df[ (df[col_name] >= df[col_name].quantile(quantile_bins[0])) & (df[col_name] <= df[col_name].quantile(quantile_bins[1]))].sample(1, random_state=random_seed)
    sample_tile_3 = df[ (df[col_name] >= df[col_name].quantile(quantile_bins[1])) & (df[col_name] <= df[col_name].quantile(quantile_bins[2]))].sample(1,random_state=random_seed)
    sample_tile_4 = df[ (df[col_name] >= df[col_name].quantile(quantile_bins[2])) & (df[col_name] <= df[col_name].quantile(quantile_bins[3]))].sample(1,random_state=random_seed)
    sample_tile_5 = df[ (df[col_name] >= df[col_name].quantile(quantile_bins[3])) & (df[col_name] <= df[col_name].quantile(quantile_bins[4]))].sample(1,random_state=random_seed)
    sample_tile_6 = df[  (df[col_name] >= df[col_name].quantile(quantile_bins[4]))].sample(1)

    stacked = pd.concat([sample_tile_1,sample_tile_2,sample_tile_3,sample_tile_4,sample_tile_5,sample_tile_6])

    fig, ax = plt.subplots(2,3, figsize=(30,30))
    fig.suptitle('Label: '+ col_name, fontsize=30, y=.02)
    plt.tight_layout(pad=4)
    row = 0
    column = 0
    for i in range(len(stacked)):
        #print(i)

        lat = stacked.iloc[i]["lat"]
        lon = stacked.iloc[i]["lon"]

        point_df = gpd.GeoDataFrame({"geometry" : [Point(lon, lat)]}, crs ="EPSG:4326")
        point_df_dst_crs = point_df.to_crs("EPSG:3857")
        point = point_df_dst_crs.loc[0, "geometry"]

        path = get_planet_tile_path(lat, lon)

        im = io.imread(path)
        raster = rasterio.open(path)
        extent = plotting_extent(raster)

        ax[row,column].imshow(im, extent=extent)  

        point_df = gpd.GeoDataFrame({"geometry" : [Point(lon, lat)]}, crs ="EPSG:4326")
        src_point = point_df.loc[0, "geometry"]
        src_point_array = np.array([src_point.x, src_point.y])
        centroid_array = np.round((src_point_array - .005), 2) + .005

        tile_df = gpd.GeoDataFrame({"geometry" : [Point(centroid_array).buffer(.005, cap_style=3)]}, crs ="EPSG:4326")
        crop_bounds = tile_df.to_crs("EPSG:3857").total_bounds

        ax[row,column].set_xlim(crop_bounds[0], crop_bounds[2])
        ax[row,column].set_ylim(crop_bounds[1], crop_bounds[3])

        ax[row,column].imshow(im, extent=extent)
        
        if show_axes == False:

            ax[row,column].axis("off")
            ax[row,column].axis("off")

        if i == 0:
            bin_string = "pecentile < "  + str(quantile_bins[0] * 100) + "%"
        elif i == 5:
            bin_string = "percentile > " + str(quantile_bins[4] * 100) + "%"
        else:
            bin_string = str(quantile_bins[i-1]*100) + "% " "< " + "percentile" + " < " + str(quantile_bins[i]*100) + "%"

        info = "lat: " + str(round(lat,3)) + ", " + "lon: " + str(round(lon,3)) + "\n" + "value: " + str(round(stacked.iloc[i][col_name],3)) + "," + "\n"
        info = info + bin_string
        #print(info)

        ax[row,column].set_title(info, size=24)

        column+=1

        if i == 2:
            row = 1
            column -= 3
        if (row == 1) & (column == 3):
            break
    if save_path:
        fig.savefig(save_path)
    
    
    