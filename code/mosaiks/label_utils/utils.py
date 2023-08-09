from pathlib import Path
from zipfile import ZipFile
import numpy as np
import math
import pandas as pd
import geopandas as gpd
import time

from shapely.ops import prep
from shapely.geometry import LineString, Point, MultiPolygon, Polygon, MultiLineString


def zip_contents(zipfile='',path_from_local=''):
    '''Prints the file names within a zipfile'''
    label_path = Path(path_from_local + zipfile)
    zf = ZipFile(label_path, 'r')
    return print([f.filename for f in zf.filelist])


def unpack_zip(zipfile='', path_from_local=''):
    '''Recursively extracts nested zipfiles in `zipfile` and saves them in the directory ``path_from_local`.'''
    
    filepath = path_from_local+zipfile
    extract_path = filepath.strip('.zip')+'/'
    parent_archive = ZipFile(filepath)
    parent_archive.extractall(extract_path)
    namelist = parent_archive.namelist()
    parent_archive.close()
    
    for name in namelist:
        try:
            if name[-4:]=='.zip':
                unpack_zip(zipfile=name, path_from_local=extract_path)
        except:
            print('failed on ',name)
            pass
    return extract_path

def num_steps(higher, lower):
    '''Calculates the number of tiles (increments of 0.01) are between a higher and lower bound. For use in the make_subgrid function'''
    return int(np.round(higher*100 - lower*100))


def make_subgrid(delta = 0.01, min_lat = -60, max_lat=74, min_lon=-180, max_lon=180):
    '''Creates a meshgrid of tile centroids, where the tiles have a resolution delta and the grid bounds are min_lat, max_lat, min_lon, max_lon. Deals with tiles that span longitude = 180/-180 degrees and data that exist entirely in a single tile.'''
    if max_lon == 180.0:
        if min_lon == max_lon:
            lon_cent = np.linspace(min_lon - delta/2, max_lon + delta/2, num = 1, endpoint = True)
        else: 
            lon_cent = np.linspace(min_lon + delta/2, max_lon + delta/2, num = num_steps(max_lon, min_lon), endpoint = False)
    else:
        if min_lon == max_lon:
            lon_cent = np.linspace(min_lon + delta/2, max_lon + delta + delta/2, num = 1, endpoint = True)
        else:
            lon_cent = np.linspace(min_lon + delta/2, max_lon - delta/2, num = num_steps(max_lon, min_lon), endpoint = True)
    
    if min_lat == max_lat:
        lat_cent = np.linspace(min_lat + delta/2, max_lat + delta + delta/2, num = 1, endpoint = True)
    else:
        lat_cent = np.linspace(min_lat + delta/2, max_lat - delta/2, num = num_steps(max_lat, min_lat), endpoint = True)

    X_cent, Y_cent = np.meshgrid(lon_cent, lat_cent)
    return X_cent, Y_cent


def nearest_tile_centroid(df, lon_col, lat_col, to_nearest=0.005):
    '''Given a Pandas dataframe containing point data identified by latitude and longitude coordinates, returns the dataframe with two new columns (x_centroid, y_centroid) indicating the coordinates of nearest grid tile centroid to each record. Inputs: the name of the dataframe (df); the columns containing the latitude and longitude coordinates (lat_col and lon_col, respectively), and the last digit of the tile centroid (to_nearest; default value 0.005 for grid tiles of 0.01 degree resolution).'''
    x_centroid = np.full(len(df),np.nan)
    for i in df.index:
        if np.isnan(df[lon_col][i])==False:
            x_centroid[i] = np.round(math.floor(df[lon_col][i]*100)/100 + to_nearest,4)
        else:
            pass
            
    y_centroid = np.full(len(df),np.nan)
    for i in df.index:
        if np.isnan(df[lat_col][i])==False:
            y_centroid[i] = np.round(math.floor(df[lat_col][i]*100)/100 + to_nearest,4)
        else:
            pass
    
    df['x_centroid'] = list(x_centroid)
    df['y_centroid'] = list(y_centroid)
    
    return df

def box_grid(grid, gpdFile, lat_col_name = "lat", lon_col_name = "lon"):
    """
    Takes a grid as an input with 'lat' and 'lon' columns and a geopandas file. 
    Isolates the grid to within the bounding box of the geopandas file. 
    
    Running this function first can be used to more efficiently check for points of interesection between a grid and a shapefile.
    """
    left, bottom, right, top = gpdFile.total_bounds
    output = grid[(grid[lon_col_name] >= left) & (grid[lon_col_name] <= right) & (grid[lat_col_name] >= bottom) & (grid[lat_col_name] <= top)]
    return output


def geopandas_shape_grid(df,gpdFile, lat_col_name = "lat", lon_col_name = "lon"):
    """
    Takes a grid df with lats and lons. The default is that these columns are named "lat" and "lon" respectively. 
    Checks for point intersections with the geopandas file that is given. 
    The geopandas file can have many MultiPolygon objects in many rows. Or it can be a simple file with one row and one polygon.
    
    Returns a new pandas df.
    
    Needs shapely, geopandas, and numpy.
    """
    gpdFile = gpdFile.copy(False) # This clears the meta data on the input dataframe. Fixes a warning.
    
    gpdFile["preped"] = gpdFile["geometry"].apply(prep) # prepare the geometry to improve speed
    
    lats = df[lat_col_name].values  #Making two arrays that together correspond to all of the grid points
    lons = df[lon_col_name].values
    
    points = [Point((lons[i], lats[i])) for i in range(len(lats))] # turn each point into a Shapely object
    
    for i, prepared_polygon in enumerate(gpdFile["preped"]):
        #print(i)
        intersect_points = list(filter(prepared_polygon.contains, points))

        if i == 0:
            hits = intersect_points
        else:
            hits = hits + intersect_points

    output_lons = []
    output_lats = []

    for i in range(len(hits)):
        output_lons.append(hits[i].x)
        output_lats.append(hits[i].y)

    outputGrid = {  
        "lat" : output_lats,
        "lon" : output_lons,
        }
    
    return pd.DataFrame(outputGrid).sort_values(["lat","lon"])



def assign_grid_points_to_gpdFile(grid, gpdFile, polygon_id_col_names):
    """
    This function efficiently returns the set of grid centroids that fall within a complex geopandas file. 
    Use the polygon_id_col_name parameter to save a corresponding polygon name or list of polygon id names to each grid point.
    
    Take a grid with "lat" and "lon" columns and a geopandas file with multiple rows.
    
    Note that per the geopandas_shape_grid() function, grid points are dropped if the centroid does not overlay any shapefile polygon
    
    See Smits_HDI.ipynb for example use case
    
    TODO refactor to remove the .loc function calls. They are VERY slow.
    
    """
    output_df = []
    counter = 0
    for i in gpdFile.index:
        counter +=1
        if counter %100 ==0:
            print(counter, "out of", len(gpdFile))

        region_gpdf = gpdFile.loc[[i]] ## subset to geodataframe to a single row
        
        region_box_grid = box_grid(grid,region_gpdf) ### isolate the full grid to a bounding box
        region_output = geopandas_shape_grid(region_box_grid, region_gpdf) ## returns all the grid centroids that fall within a certain region
        
        if type(polygon_id_col_names) == str:
            region_output["unique_polygon_id"] = gpdFile.loc[i,polygon_id_col_names]
            
        elif type(polygon_id_col_names) == list: #allows us to pass a list of polygon id column names. E.g., we can save a region code and a country code
            num = 0
            for j in polygon_id_col_names:
                name = "polygon_id" + str(num)
                region_output[name] = gpdFile.loc[i,j]
                num += 1
                
        else:
            print("polygon_id_col_names parameter must be a string or a list")
            raise Exception
        
        output_df.append(region_output)
        
    return pd.concat(output_df)



def create_subgrid_for_country(country_code, base_grid_path="/shares/maps100/data/output/grid/LandmassIntermediateResSparseGrid10.csv"):
    
    """ 
    Takes a two character country code as an input and outputs a landmass isolated subgrid for the country. 
    
    By default, this function reads data from the LandmassIntermediateResSparseGrid10.
    
    Try:
    
    create_subgrid_for_country("US)
    
    """
    grid = pd.read_csv(base_grid_path)
    
    country_path = "/shares/maps100/data/raw/country_bounds/ne_10m_admin_0_countries.shp"
    countries = gpd.read_file(country_path)
    country_gpd = countries[countries["ISO_A2"] == country_code]
    
    left_bound, bottom_bound, right_bound, top_bound = country_gpd.total_bounds

    indices = (grid["lon"] >= left_bound) & (grid["lon"] <= right_bound)  & (grid["lat"] <= top_bound) & (grid["lat"] >= bottom_bound)
    box_grid = grid.loc[indices]
    country_grid = geopandas_shape_grid(box_grid,country_gpd)
    
    return country_grid


def create_subgrid_for_state(state_name, base_grid_path="/shares/maps100/data/output/grid/LandmassIntermediateResSparseGrid10.csv"):
    """ 
    Takes a US state name as an input and returns a grid that is isolated to that state. 
    
    The default base grid is LandmassIntermediateResSparseGrid10.csv
    
    Try:
    
    create_subgrid_for_state("California")
    """
    grid = pd.read_csv(base_grid_path)
    
    state_path = "/shares/maps100/data/raw/us_bounds/USA_adm1.shp"
    states = gpd.read_file(state_path)
    state_gpd = states[states["NAME_1"] == state_name]
    
    left_bound, bottom_bound, right_bound, top_bound = state_gpd.total_bounds

    indices = (grid["lon"] >= left_bound) & (grid["lon"] <= right_bound)  & (grid["lat"] <= top_bound) & (grid["lat"] >= bottom_bound)
    box_grid = grid.loc[indices]
    state_grid = geopandas_shape_grid(box_grid,state_gpd)
    
    return state_grid


def get_shapefile_dense_grid_centroids(gpdf, columns):
    """
    Takes a one row geopandas dataframe corresponding to a single shapefile polygon. 
    
    It finds the dense grid centroids that fall within that polygon and returns them.
    
    The columns parameter specifies the set of columns to keep in the output
    """
    min_lon, min_lat, max_lon, max_lat = np.round(np.array(gpdf["geometry"].iat[0].bounds),2)
    lons,lats = make_subgrid(.01,min_lat,max_lat,min_lon,max_lon)
    grid = pd.DataFrame({"lat" : lats.flatten(), "lon":lons.flatten()})
    output_grid = geopandas_shape_grid(grid,gpdf)
    
    for column in columns:
        output_grid[column] = gpdf[column].iat[0]
    
    return output_grid

def get_dense_grid_for_gpdf_file(gpdf, columns):
    """
    Takes a multi-row geopandas shapefile as an input. 
    
    It returns all the grid centroids that fall within the file. The output dataframe also has the relevant polygon column information for each point.
    
    TODO refactor to remove the .iloc function call. It is VERY slow.
    """
    out = []
    
    for i in range(len(gpdf)):
        if i % 100 == 0:
            print(i, " out of ", len(gpdf))
        grid = get_shapefile_dense_grid_centroids(gpdf.iloc[[i]],columns)
        out.append(grid)
        
    return pd.concat(out,ignore_index=True)