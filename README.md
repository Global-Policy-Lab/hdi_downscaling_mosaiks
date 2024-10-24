# HDI Downscaling
Replication code for ["Global High-Resolution Estimates of the United Nations Human Development Index Using Satellite Imagery and Machine Learning"](https://www.nber.org/papers/w31044) which is currently available as an NBER Working Paper.

Contact Luke Sherman via lsherm@stanford.edu with questions.

### Note on replication
Full replication of this research effort requires a large number of external data sources and considerable computing effort. Even for basic replication, it will be necessary for users to download a number of these external data products and manually add them to the user's data directory. In the replication instructions below, we will be as detailed as possible about how to do this. 

## Structure

The core analyses are from this research effort are available in jupyter notebooks in the `code/analysis/` directory. The most relevant notebook files are as follows:

- `hdi_and_iwi_model_training.ipynb` - This notebook is where HDI, and IWI ridge models are trained. Executing this notebook requires the MOSAIKS daytime image features for the ADM1 and ADM0 polygons, which are included in the code repository. The NL features are available in this repo, and their creation can also be replicated by downloading the raw VIIRS data. 
- `hdi_preds_at_adm2.ipynb` - In this notebook, we generate HDI predictions from the ADM1 models at the ADM2 level. MOSAIKS and NL features are needed at the ADM2 level to create these predictions. The MOSAIKS features for ADM2 polygons must be downloaded from [api.mosaiks.org](https://api.mosaiks.org/portal/precomputed).
- `hdi_preds_at_grid_viirs.ipynb` - In this notebook, we generate HDI predictions from the ADM1 models at the 0.1 x 0.1 degree grid level. Running this notebook requires the MOSAIKS features at the 0.1 x 0.1 degree grid level (48GB). For replication, the grid data can be downloaded in chunks from `https://siml.berkeley.edu/portal/global_grids/`. 
- `mexico_downscaling_analysis_ADM2.ipynb` - In this notebook, we compare ADM2 HDI predictions to the data from [Permanyer (2013)](https://www.sciencedirect.com/science/article/abs/pii/S0305750X1200294X) who calculates HDI estimates at the ADM2 level using census data. 
- `iwi_downscaling_analysis_dhs_cluster.ipynb` - In this notebook, predictions of the International Wealth Index (IWI) are generated and evaluated at the DHS cluster level. While we make this code available, replicating this analysis requires DHS cluster level locations and information that are not publicly available. Please contact me via (lsherm@stanford.edu) if you would like to get access to this data. I will coordinate with the DHS program and the Global Data Lab to see if providing access is possible. 
- `nl_downscaling_experiment_ADM2.ipynb` - In this notebook, we train a MOSAIKS model to predict NL at the ADM1 level and then evaluate performance at the ADM2 level. Additional MOSAIKS features need to be downloaded to replicate this analysis.
- `hdi_label_creation.ipynb` - This file is used to process the HDI source data from the Global Data Lab. It can be used to replicate the data cleaning steps.
- `prediction_utils.py` - This file contains extensive functionality used for ridge model solves, hyperparameter tuning, and data cleaning. A few of these functions contain dependencies in the `code/mosaiks/` directory. 


## Demo
For convenience, we have created a standalone demonstration notebook that can be run in approximately 5 minutes on a laptop computer. This notebook trains a model to predict HDI at the ADM1 level and is more simple than the models used in the primary analysis. All data files needed to run this demo are included in this GitHub repository. The demo notebook can be found at `code/demo//DEMO_simplified_HDI_model_train.ipynb`.


## Replication instructions

Users aiming to replicate this research will likely need a basic familiarity with python, anaconda, and Jupyter. Here we offer replication instructions using anaconda. The core analysis are displayed in Jupyter Notebook files (`.ipynb` files). 

### Basics
1. To replicate this analysis, users should start by cloning this github repository to their local machine. 

2. Users will need to create a conda environment based on the `environment/req.txt` file. Navigate to this folder on the command line and run `conda create --name hdi_downscaling --file req.txt`. If you are not using a linux machine, the conda environment will need to be created from the `req.yml` file. Installing the conda environment can take between 30 minutes to an hour.

3. Users must set the github repo directory as an environment variable called `REPO_DIR`. For me, this involves running the following at the command line `$ export REPO_DIR=/home/lsherman/code_luke/hdi_downscaling_mosaiks/`. You must replace the path with your own local path to the cloned repository. Be sure to include the trailing `/` character. To ensure that this has been set, use `$ echo $REPO_DIR` at the command line and ensure the repo directory is returned.


### MOSAIKS Features
4. MOSAIKS features at ADM1 and ADM0 (model training) are included in the repo. MOSAIKS features for ADM2 polygons must be downloaded directly at [api.mosaiks.org/portal/precomputed](https://api.mosaiks.org/portal/precomputed/). To replicate this analysis, download the ADM2 population weighted python/pickle file and add it to the `/data/features/mosaiks_features/` directory. Note that these aggregated MOSAIKS features can also be used to predict a variety of other tasks at the ADM level of observation.

### Shapefiles.
5. Download ADM2 shapefile  from [geoBoundaries (CGAZ 3.0.0)](https://www.geoboundaries.org/data/geoBoundariesCGAZ-3_0_0/ADM2/simplifyRatio_100/geoBoundariesCGAZ_ADM2.geojson). This file should now be named `data/raw/geoBoundaries/geoBoundariesCGAZ_ADM2.geojson`. 
6. Download ADM1 shapefile V6.1 from [Global Data Lab](https://globaldatalab.org/mygdl/downloads/shapefiles/). These files should be unzipped and placed in this directory: `data/raw/GDL_HDI/`.
7. Mexico's ADM2 shapefile is also required to create the crosswalk with the geoBoundaries ADM2 regions. This is available for download from the [World Bank](https://datacatalog.worldbank.org/search/dataset/0039294). The shapefile should be unzipped and placed in the following directory: `data/undp/raw/Permanyer_MHDI/muni_2012gw/`.

### NL Data
6. For all analyses that use nightlight luminosity, we use a single VIIRS data product. We use the annual 2019 VIIRS composite V2.1 average masked data product. This can be [downloaded here](
https://eogdata.mines.edu/nighttime_light/annual/v21/2019/). This data is only required for users who want to replicate our processing of the raw NL data product. The processed features and average values are included in this repo.

### Population data
8. For population weighting of the MOSAIKS and NL features, and for the population estimates show in Figure 4 we use population data from Global Human Settlement Data Layer (GHS-POP). We use the estimate for the year 2020 at the 30 arcsecond resolution. This data is [available here](https://ghsl.jrc.ec.europa.eu/download.php?ds=pop). 


### Spatial cross walks
9. To link shapefiles, we create a few crosswalk files. This is done in the `code/analysis/shapefile_crosswalk_creation` directory. We include the code needed to generate these files.
