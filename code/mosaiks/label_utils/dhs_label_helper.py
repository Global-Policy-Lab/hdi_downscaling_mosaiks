import numpy as np
import pandas as pd
import os
import geopandas as gpd

def dhs_pct_by_cluster(num_df, den_df, var, aggfunc,cluster_col='hv001'):
    '''
    Inputs:
        num_df: a dataframe containing the subset of the DHS survey representing the numerator of the label (e.g., households that have access to electricity).
        den_df: a dataframe containing the subset of the DHS survey representing the denominator of the label (e.g., all households).
        var: the name of the column in the DHS survey that holds the outcome of interest.
        aggfunc: specify as "sum" or "count". Specifies how numerators and denominators should be aggregated. 
                If "sum," values in the var column are summed for each cluster. If "count," the number of observations are 
                counted for each cluster.
        cluster_col: the name of the column in the DHS survey that holds the cluster identifier.
    Output: a dataframe in which each row represents a cluster (column "DHSCLUST". Columns include the numerator ("num"), denominator ("den"),
            and "percent" (num/den * 100).

    '''
    if aggfunc=='sum':
        num = num_df.groupby(cluster_col,as_index=False).sum()[[cluster_col,var]]
        den = den_df.groupby(cluster_col,as_index=False).sum()[[cluster_col,var]]
    elif aggfunc=='count':
        num = num_df.groupby(cluster_col,as_index=False).count()[[cluster_col,var]]
        den = den_df.groupby(cluster_col,as_index=False).count()[[cluster_col,var]]
    
    num.rename(columns={var:'num'},inplace=True)
    den.rename(columns={var:'den'},inplace=True)
    clusts = pd.merge(num,den,how='outer',on=cluster_col)
    clusts['percent'] = clusts['num']/clusts['den']*100
    clusts.rename(columns={cluster_col:'DHSCLUST'},inplace=True)
    clusts.loc[clusts['percent'].isnull(),'percent']=0 #Replace null values with zero (a null value means that there was no numerator column associated with the denominator column, i.e., numerator = 0)
    return clusts


def indicator_df(survey_list,survey,indicator_func,survey_columns,extraArgs={},cutoff_yr=2012,categoricals=True, path='/shares/maps100/data/raw/applications/DHS/'):
    '''
    Inputs:
        survey_list: label-specific list of the latest DHS surveys for which both the outcome variable and GIS data are available.
        survey: DHS survey recode type. Can be, for instance, "HR" (household recode), "PR" (hosuehold members recode), "BR" (births recode)
        indicator_func: the name of the function that processes DHS survey data into cluster-specific outcomes. Defined separately for each label.
        survey_columns: a list of columns from the DHS survey to include in the analysis. 
        extraArgs: a dictionary defining any additional variables that need to be specified for the indicator_func.
        cutoff_yr: the earliest DHS survey year to include in the label. Default is 2012.
        categoricals: specify True to convert DHS survey data to categorical values; False to use numerically encoded values.
        path: file path to directory holding all raw DSH survey data.
    Output: a dataframe that includes cluster-specific labels (outcomes) and their corresponding GIS coordinates for all relevant DHS surveys.
    '''
    all_countries = pd.DataFrame({}) # create an empty df to store data created through each iteration
    
    for cn_path in survey_list: # iterates through each survey in survey_list
        if int(cn_path[3:7])<cutoff_yr: # exclude surveys conducted before the cutoff year
            continue
        print(cn_path)
        for f in os.listdir(path + cn_path): # each DHS survey is associated with several subfolders
            if survey in f: # we want the subfolder corresponding to the `survey` input
                print('processing {} recode'.format(survey))
                pref = f.partition('DT')[0]
                df = pd.read_stata(path + cn_path + '/'+ f + '/' + pref + 'FL.DTA',  # reads the survey data file
                                   columns=survey_columns, # processes only those columns needed for the label analysis 
                                   convert_categoricals=categoricals) # converts column values to categoricals, or not
                indicator = indicator_func(df,**extraArgs) # runs the indicator-specific function for the country specific dataset
            elif 'GE' in f: # reads the GIS cluster data associated with the survey
                print('processing GIS')
                gdf = gpd.read_file(path + cn_path + '/'+ f + '/' + f + '.shp')
            else:
                pass
        label_gis = pd.merge(indicator,gdf,on='DHSCLUST', how='inner') # merges the cluster-specific outcomes with the cluster's GIS coordinates
        label_gis.drop(label_gis.loc[(np.round(label_gis['LONGNUM'])==0)&(np.round(label_gis['LATNUM'])==0)].index,inplace=True) # drop records with null coordinates
        all_countries = pd.concat([all_countries,label_gis],ignore_index=True) # add this survey's spatially explicit outcome to the running list
        
    return all_countries


def finish_label(df, tiles,label_col='percent'):
    '''
    Inputs: 
        df: the dataframe containing the cluster-specific label outcomes and GIS coordinates. Needs to include a "DHSID" column.
        tiles: master df of all DHS tiles.
        label_col: the name of the column of `df` that holds the label value.
    Outputs: a dataframe containing the tile latitude, longitude, outcome, and associated DHSID for all tiles associated with the clusters included in a particular label. 
    '''
    # get rid of extraneous columns 
    df_reduced = df[[label_col,'DHSID']] 
    tiles_reduced = tiles.loc[:,['lon','lat','DHSID','surveyID','regionID']] 
    # merge tiles with cluster-specific outcomes based on DHSID
    label_tiles = pd.merge(df_reduced,tiles_reduced,how='left',on='DHSID')
    return label_tiles[['lon','lat',label_col,'DHSID','surveyID','regionID']]


def test_single_survey(survey_list,survey_index,survey,survey_columns,cutoff_yr=2012,categoricals=True, path='/shares/maps100/data/raw/applications/DHS/'):
    '''
    Function that creates a survey dataframe on which to test new indicator functions.
    '''
    
    cn_path = survey_list[survey_index]
    print(cn_path)
    for f in os.listdir(path + cn_path):
        if survey in f:
            pref = f.partition('DT')[0]
            df = pd.read_stata(path + cn_path + '/'+ f + '/' + pref + 'FL.DTA', 
                               columns=survey_columns, 
                               convert_categoricals=categoricals)
        elif 'GE' in f:
            print('processing GIS')
            gdf = gpd.read_file(path + cn_path + '/'+ f + '/' + f + '.shp')
        else:
            pass
    return df