"""This module is loaded to access all model settings.
The first group contains global settings, and then each
application area has its own settings. The variable name for
each application must be associated with a dictionary that
at the minimum has the following keys:


Keys:
-----
application : str
    Name of application area.
variable : str or list of str
    The name(s) of the variable(s) being predicted in this
    application area.
sampling : str
    The sampling scheme used for the preferred model for this
    application area (e.g. 'POP')
"""
######################
# PROJECT SETTINGS
######################

# Verbose
verbose = True

# GPU -- set to True to use torch.cuda.is_available to check for GPUs.
# If False, will never check for a GPU.
GPU = False

import os
from os.path import basename, dirname, join

import numpy as np

import seaborn as sns
import mosaiks 
import matplotlib.colors as mcolors

features_dir = "/shares/maps100/data/features/"

# shapefile paths (for regional models)
# continent
shp_path_stem = "/shares/maps100/data/raw/region_shapefiles/"
continent_shp_path = (shp_path_stem +
                      "World_Continents_NoAntarctica/" +
                      "World_Continents_NoAntarctica.shp")

# figures directory for final paper
main_fig_dir = "/shares/maps100/results/figures"    

# country
country_shp_path = (shp_path_stem +
                    "gadm/gadm36_0.shp")

# path to sparse grid pop density file; for weighted feature aggregation 
sparse_grid_pop_density_path = None

dense_grid_pop_density_path = ('/shares/maps100/code/code_LS/hdi_downscaling/'
    'data/int/GHS_pop/ghs_pop_rcf_weights_.01x.01.p')

pop_density_val_colname = "pop_weights"

# ML MODEL
ml_model = {
    "seed": 0,
    "test_set_frac": 0.2,
    "model_default": "ridge",
    "n_folds": 5,
    "global_lambdas": np.logspace(-4, 3, 9),
}


# colors
colorblind_friendly_teal = "#029e73"


# REGIONAL MODELS
continent_vm_dict = {
    'Africa': ['4', '5', '6', '7'],
    'Asia': ['1', '4', '5', '6', '7', '8', '9', '10'],
    'Australia': ['9', '10'],
    'North America': ['1', '2', '3'],
    'Oceania': ['1', '9', '10'],
    'South America': ['2', '3', '4'],
    'Europe': ['4', '5', '6']
}

shp_file_dict = {
    "continent": continent_shp_path,
    "country": country_shp_path
}

region_col_dict = {
    "continent": "CONTINENT",
    "country": "NAME_0"
}


# Default thresholds for Hurdle model:
default_hurdle_thresholds = [0.95, 0.90, 0.8, 0.65, 0.5]


# Category colors
category_colors = {'Demographics': '#4E79A7',
                'Education': '#F28E2B',
                'Health': '#E15759',
                'Income': '#76B7B2',
                'Occupation': '#59A14F',
                'Household Assets':  "#D4AF37",
                'Agricultural Assets': '#A020F0',
                'Agriculture': '#EC008C',
                'Built Infrastructure': '#161616',
                'Natural Systems': '#964B00'
                }
# Category cmaps
category_cmaps = category_colors.copy()
for key in category_cmaps.keys():
    category_cmaps[key] = mcolors.LinearSegmentedColormap.from_list("", ["white", category_cmaps[key]])

