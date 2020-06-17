# USA Building Height Inference

## Introduction
The code in this repository was developed during the graduation project of Imke Lansky for the MSc Geomatics at TU Delft, the Netherlands (for more details, the thesis is available [here: to be added]).

The [USBuildingFootprints](https://github.com/microsoft/USBuildingFootprints) dataset, designed by Microsoft, is used to predict the building heights for. All footprints are characterised by nine geometric features that are derived from the footprints: *area, compacntess, complexity, length, width, slimness, number of neighbours, number of adjacent buildings,* and *number of vertices*. Furthermore, a distinction is created between two types of area morphologies: the Central Business Districts (CBDs) and the more rural and suburban areas. Based on this classification, either a split prediction model can be used or a single prediction model. The former creates a prediction model for each area morphology, while the latter only creates one prediction model and uses the area morphology as an additional feature in the training and prediction process.

## Dependencies
The building height inference program is written in `Python (v3.7.7)`. For storing data, a `PostgreSQL (v11.6)` database extended with `PostGIS (v2.5.2)` is used. The implementation is dependent on the following libraries:

* [fiona v1.8.13.post1](https://pypi.org/project/Fiona/)
* [geojson v2.5.0](https://pypi.org/project/geojson/)
* [geopandas v0.7.0](https://pypi.org/project/geopandas/)
* [joblib v0.14.1](https://pypi.org/project/joblib/0.14.1/)
* [matplotlib v3.2.1](https://pypi.org/project/matplotlib/)
* [numpy v1.18.4](https://pypi.org/project/numpy/1.18.4/)
* [pandas v1.0.3](https://pypi.org/project/pandas/1.0.3/)
* [psycopg2-binary v2.8.5](https://pypi.org/project/psycopg2-binary/)
* [scikit-learn v0.23.0](https://pypi.org/project/scikit-learn/0.23.0/)
* [seaborn v0.10.1](https://pypi.org/project/seaborn/)
* [shapely v1.7.0](https://pypi.org/project/Shapely/)

## Structure
The repository is structured as follows:
* **/shell_scripts**: shell scripts that were written for some data processing and loading the building footprints to the database.
* **/code**: the Python scripts that contain the machine learning implementation and some pre-processing steps.


## Usage
* **Data preparation**: `add_unique_ID.py` can be used to read the GeoJSON files in a given folder and provide each footprint with a unique identifier.
* **Feature extraction**: `extract_features.py` is used to set up a connection to the database and reproject the coordinates of the footprints and then extract the features.
* **Tuning**: `cross_validation.py` is used to run k-fold cross validation with a randomised grid search to tune the hyperparameters of the models based on the training data.
* **Height predictions**: `infer_heights.py` is run to start the building height inference process. It utilises the parameters defined in `params.json`, see the section below.
* **Analyse**: `analyse_results.py` is used to analyse the results that are stored in a file (not in a database).
* **Additional features**: `heights_non_geom_features.py` is used to analyse a separate test area for the city of Denver, where the *population density, average household income, average household size, building type, number of amenities,* and *raster building height* are used as additional features.
* **Generate 3D city model**: `extruder.py` can be used to create a 3D city model from 2D building footprints that have the building height stored as an additional attribute. Source: [here](https://github.com/cityjson/misc-example-code/blob/master/extruder/extruder.py), slightly adapted for this use-case.

### Parameters
The `params.json` file contains the parameters that can be set by the user. The options are the following:


* `method`: `RFR/MLR/SVR`. The machine learning method to use.
* `db_traindata`: The name of the database that contains the training data. If left empty, the user can provide a prediction model that is saved to a file.
* `tables_traindata`: List containing the tables to use in the training database. If left empty, all public tables will be used.
* `db_testdata`: The name of the database that contains the test data. If left empty, only training of the prediction model is performed.
* `tables_testdata`: List containing the tables to use in the testing database. If left empty, all public tables will be used.
* `labels_for_testdata`: `true/false`. Whether the data in that is used for testing has the labels (i.e. building height) available or not. If available, error metric for the results are provided.
* `network_type`: `single/split`. The type of prediction model to use. `single` means one model is used with the area morphology as an additional feature. `split` uses two models, one for each area morphology.
* `save_predictions`: `true/false`. Whether to store the building height predictions in the database or not.
* `save_prediction_model`: `true/false`. Whether to save the trained prediction model in a file on disk.
* `test_subsets`: `true/false`. Whether to train and test the models with only a subset of the possible features.
* `feature_subset`: List containing the features that should be used in the subset test.
* `generate_plots`: `true/false`. Whether to create plots based on the training data.
* `model_suburbs`: Path to the model file to use for the suburban and rural data if no training database is specified.
* `scaler_suburbs`: Path to the scaler file to use for the suburban and rural data if no training database is specified.
* `model_cbds`: Path to the model file to use for the CBD data if no training database is specified.
* `scaler_cbds`: Path to the scaler file to use for the CBD data if no training database is specified.
* `model_single_network`: Path to the model file to use for the combined data if no training database is specified.
* `scaler_single_network`: Path to the scaler file to use for the combined data if no training database is specified.
