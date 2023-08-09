import pandas as pd
import numpy as np
import math



def make_label(df,uniqueTiles,labelName,labelCol, aggfunc):
    '''Given a dataframe and a set of unique tile centroids within the dataframe, returns a new dataframe containing with columns for the longitude, latitude, and label value for each tile. This function aggregates data based on the input 'aggfunc', which can either be 'unique' (in which case data are aggregated into tiles by counting the unique values of the labelCol associated with each tile centroid) or 'max' (in which case the tile is assigned the maximum value of the data).'''
    lon = []
    lat = []
    label = []
    
    for t in uniqueTiles.index:
        t_lon = uniqueTiles['x_centroid'][t]
        t_lat = uniqueTiles['y_centroid'][t]
        lon.append(t_lon)
        lat.append(t_lat)
        
        df_t = df.loc[(df['x_centroid']==t_lon)&(df['y_centroid']==t_lat)]
        
        if aggfunc == 'unique':
            label.append(len(df_t[labelCol].unique()))
            
        elif aggfunc == 'max':
            label.append(np.nanmax(df_t[labelCol]))
        
    label_df = pd.DataFrame({'lon': lon, 'lat':lat, labelName:label})
    
    return label_df