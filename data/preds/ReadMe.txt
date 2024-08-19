 # ReadMe.txt # 
 
 
Global High Resolution Estimates of the Human Development Index, Version 2.0 

## BACKGROUND
This `ReadMe` file provides information necessary to understand and interpret the accompanying 
raster (.tif) and tabular (.csv) HDI data files.

Note that these data should be used cautiously. These are only estimates and may contain
noise, systematic bias, compressed variance and other errors. 


GitHub github.com/Global-Policy-Lab/hdi_downscaling_mosaiks/
Email: lsherm@stanford.edu



## RASTER DATA
Layer 1: 		HDI Grid Level Estimates
					- These are the predictions generated from the machine learning model 
					  combining MOSAIKS daytime image and nightlight (NL) features
					  
					- These predictions are clipped to have values that fall within 0 and 1.
					
					- These estimates are centered such that the population-weighted average
					   of the 0.01 x 0.01 degree tile HDI estimates matches the ADM1 data
					   from Smits and Permanyer (https://globaldatalab.org/shdi).
					   
					- Population weights come from the Global Human Settlement 
					  Data (GHS-POP) (https://ghsl.jrc.ec.europa.eu/download.php?ds=pop).
					  
					- Note that we only release estimates where human settlements are known to
					  exist. 


resolution: 		0.1 x 0.1 degree
no data value:		numpy.nan
raster extent:		[-180,180, -56, 74]





## TABULAR DATA
Tablular data can be merged with the ADM2 administrative geojson (CGAZ V3.0.0) from geoBoundaries 
available here: 
https://www.geoboundaries.org/data/geoBoundariesCGAZ-3_0_0/ADM2/simplifyRatio_100/geoBoundariesCGAZ_ADM2.geojson

Columns:
predicted_adm2_HDI: 			- Predictions generated from the machine learning model 
					  		   combining MOSAIKS daytime image and nightlight (NL) features
					  		   
					  		- Predictions are centered such that the average
					   		  of the ADM2 estimates contained by each ADM1 region matches
					   		  the ADM1 value from Smits and Permanyer (https://globaldatalab.org/shdi).
					   		
					   		- Estimates are not released where Smits and Permanyer have not
					   		  produced ADM1 HDI estimates.
					   		  
					   		- ADM2 estimates for Ireland are not released as the ADM2 units 
					   		  from geoBoundaries are so small that they are not cannot be
					   		  adequately verified.
					   		  
est_total_pop:				- Total population count contained in each ADM2 region. This 
					data comes from GHS-POP.
							  
adm1_HDI_Smits:				- 2019 HDI estimate for the parent Global Data Lab
					ADM1 administrative unit. Please cite Smits and Permanyer (2019)
					https://www.nature.com/articles/sdata201938 if using these data.
							  
percent_overlap_GDL_ADM1 		- Indicates the percent of the ADM2 geoBoundaries ADM2 unit
					that overlaps the Global Data Lab ADM1 shapefile (using a 
					WGS84 projection). Lower overlap implies more uncertainty 
					n the re-centering.

GDL_ADM1				- ADM1 code used by the Global Data Lab (GDL) and can be
					used to merge with their data. 
							  
ADM1_shapeID				- Parent ADM1 code used by geoBoundaries. Note that
					this is not an exact match with the GDL shapefile

shapeGroup				- Standard ISO3 code identifying the parent country

shapeName				- Common name for the ADM2 unit from geoBoundaries

shapeID					- Unique identifier for each ADM2 unit from geoBoundaries
							
	
	
