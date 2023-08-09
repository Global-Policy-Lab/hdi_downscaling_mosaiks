
from mosaiks.utils.logging import log_text
from mosaiks.utils import io
from mosaiks.solve import data_parser as parse
from mosaiks.plotting import general_plotter as plots
from mosaiks import config as c

from mosaiks.utils import config_read

from mosaiks.solve import master_solve as ms


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




def prediction_wrapper(app, from_meta=False, c=c, return_dict=False,
                    save_plots=True, subdirectory=False):
    """
    Prepare the environment, load in a config, prepare data, train a MOSAIKS 
    model, and produce standard outputs.

    Args:
            app: the name of your task (must have a config of the same name in
                mosaiks/configs)
        return_dict: boolean. Return a dictionary containing all the information
            needed to reconstruct any part of the analysis outside of the wrapper.
        subdirectory: False or str
            Look for config in subfolder of config directory. Only applies when from_meta = False.
            Outputs will be saved in subdirectory of same name
    """

    (c, c_app, outcome_name, labels_file, grid,
        polygon_id_colname) = config_read.extract_config(app, c=c, from_meta =from_meta, subdirectory=subdirectory
                                                )

    log_text('Config parsed.')

    X_train, Y_train, locations_train, regions_train = prep_data(
        c, app, c_app, polygon_id_colname,labels_file
        )


    log_text('Data loaded and split.')


    kfold_results = ms.master_kfold_solve(c, c_app, X_train,
                                                Y_train, 
                                                locations_train,
                                                regions_train)

    log_text('Model trained.')


    plots.plot_diagnostics(kfold_results, outcome_name, app,
                    polygon_id_colname, grid, c, c_app,
                        save=save_plots)

    if return_dict:
        names = ("c", "c_app", "outcome_name", "labels_file", "grid",
                    "polygon_id_colname", "X_train", "Y_train", 
                    "locations_train", "regions_train", "kfold_results")
        
        vals = (c, c_app, outcome_name, labels_file, grid,
                    polygon_id_colname, X_train, Y_train,
                locations_train, regions_train, kfold_results)
        
        dct = {names[i]: vals[i] for i in range(len(vals))}

        return dct
