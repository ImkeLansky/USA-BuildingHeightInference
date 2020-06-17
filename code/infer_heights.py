"""
Perform different machine-learning approaches on the footprint data.
"""

from time import time
from os import path
import sys
from datetime import timedelta
import json
import numpy as np
import db_funcs
import ml_funcs
import generate_plots


def train_from_data(method, train_features, train_labels, test_features, save_model, net_type):
    """
    Perform the machine learning for the selected method.
    The prediction model is trained based on the training data.
    """

    if method == "RFR":
        # If there are test features, more information is returned.
        if len(test_features) != 0:
            predictions, importances, _ = ml_funcs.rf_from_traindata(train_features, train_labels,
                                                                     test_features, net_type,
                                                                     save_model)
            return predictions, importances

        if len(test_features) == 0:
            importances, _ = ml_funcs.rf_from_traindata(train_features, train_labels,
                                                        test_features, net_type, save_model)
            return importances

    if method == "MLR":
        if len(test_features) != 0:
            predictions = ml_funcs.mlr_from_traindata(train_features, train_labels, test_features,
                                                      net_type, save_model)
            return predictions

        if len(test_features) == 0:
            ml_funcs.mlr_from_traindata(train_features, train_labels, test_features, net_type,
                                        save_model)

    elif method == "SVR":
        if len(test_features) != 0:
            predictions = ml_funcs.svr_from_traindata(train_features, train_labels, test_features,
                                                      net_type, save_model)
            return predictions

        if len(test_features) == 0:
            ml_funcs.svr_from_traindata(train_features, train_labels, test_features, net_type,
                                        save_model)

    else:
        print("The selected method is not valid! Choose from: <RFR>, <MLR> or <SVR>")
        sys.exit()


def predict_from_model(method, test_features, model, scaler, net_type):
    """
    Perform the machine learning for the selected method.
    No training is performed, the model is retrieved from a file.
    """

    if method == "RFR":
        predictions, importances = ml_funcs.rf_from_model(test_features, model, scaler, net_type)
        return predictions, importances

    if method == "MLR":
        predictions = ml_funcs.mlr_from_model(test_features, model, scaler, net_type)
    elif method == "SVR":
        predictions = ml_funcs.svr_from_model(test_features, model, scaler, net_type)
    else:
        print("The selected method is not valid! Choose from: <RFR>, <MLR> or <SVR>")
        sys.exit()

    return predictions


def split_network(train_suburbs, train_cbds, test_db, tables_test, model_suburbs, scaler_suburbs,
                  model_cbds, scaler_cbds, method, save_predictions, labels, save_model,
                  test_subsets, feature_subset):
    """
    Perform the machine learning based on a split training network
    that separates the CBDs and Suburban and Rural areas.
    """

    reading_time = []
    predict_time = []

    if len(train_suburbs) != 0:
        train_feat_suburb, train_label_suburb = ml_funcs.get_features_and_labels(train_suburbs,
                                                                                 "split",
                                                                                 test_subsets,
                                                                                 feature_subset,
                                                                                 labels=True)

    if len(train_cbds) != 0:
        train_feat_cbd, train_label_cbd = ml_funcs.get_features_and_labels(train_cbds, "split",
                                                                           test_subsets,
                                                                           feature_subset,
                                                                           labels=True)

    # A database to perform the tests/ predictions on is specified.
    if test_db:
        connection = db_funcs.setup_connection(test_db)
        connection.autocommit = True
        cursor = connection.cursor()

        # If no specific tables are selected, perform predictions
        # for all tables in the specified testing database.
        if not tables_test:
            tables_test = db_funcs.unique_tables(cursor)

        for table in tables_test:
            if table == 'cbds':
                continue

            print(80*'-')
            print(80*'-')

            starttime = time()

            test_suburbs, test_cbds, _ = db_funcs.read_data(connection, table, training=labels)

            endtime = time()
            duration = endtime - starttime
            reading_time.append(duration)

            if labels:
                test_feat_suburbs, test_labels_suburbs = \
                  ml_funcs.get_features_and_labels(test_suburbs, "split", test_subsets,
                                                   feature_subset, labels=labels)
                test_feat_cbds, test_labels_cbds = \
                  ml_funcs.get_features_and_labels(test_cbds, "split", test_subsets, feature_subset,
                                                   labels=labels)
            else:
                test_feat_suburbs = \
                  ml_funcs.get_features_and_labels(test_suburbs, "split", test_subsets,
                                                   feature_subset, labels=labels)
                test_feat_cbds = ml_funcs.get_features_and_labels(test_cbds, "split", test_subsets,
                                                                  feature_subset, labels=labels)

            pred_cbds, pred_suburbs = np.array([]), np.array([])

            starttime = time()

            # There is no training data specified, use model.
            if len(train_suburbs) == 0 and len(train_cbds) == 0:
                if method == "RFR":

                    # There must be test features for the CBD present.
                    if len(test_feat_cbds) != 0:
                        pred_cbds, imp_cbds = predict_from_model(method, test_feat_cbds,
                                                                 model_cbds, scaler_cbds,
                                                                 'CBD')
                    else:
                        print("Warning: no CBD data present in test set {0}".format(table))

                    # There must be test features for the suburbs/rural areas present.
                    if len(test_feat_suburbs) != 0:
                        pred_suburbs, imp_suburbs = predict_from_model(method, test_feat_suburbs,
                                                                       model_suburbs,
                                                                       scaler_suburbs, 'suburbs')
                    else:
                        print("Warning: no rural/suburban data present in test set {0}"\
                              .format(table))

                else:
                    # There must be test features for the CBD present.
                    if len(test_feat_cbds) != 0:
                        pred_cbds = predict_from_model(method, test_feat_cbds, model_cbds,
                                                       scaler_cbds, 'CBD')
                    else:
                        print("Warning: no CBD data present in test set {0}".format(table))

                    # There must be test features for the suburbs/rural areas present.
                    if len(test_feat_suburbs) != 0:
                        pred_suburbs = predict_from_model(method, test_feat_suburbs,
                                                          model_suburbs, scaler_suburbs,
                                                          'suburbs')
                    else:
                        print("Warning: no rural/suburban data present in test set {0}"\
                              .format(table))

            # There is training data specified, check which area morphologies are present.
            else:
                if method == "RFR":
                    if len(train_suburbs) != 0 and len(test_feat_suburbs) != 0:
                        pred_suburbs, imp_suburbs = train_from_data(method, train_feat_suburb,
                                                                    train_label_suburb,
                                                                    test_feat_suburbs,
                                                                    save_model, 'suburbs')
                    else:
                        print("Warning: training and testing data do not both contain " +\
                              "suburban/rural data!")

                    if len(train_cbds) != 0 and len(test_feat_cbds) != 0:
                        pred_cbds, imp_cbds = train_from_data(method, train_feat_cbd,
                                                              train_label_cbd, test_feat_cbds,
                                                              save_model, 'CBD')
                    else:
                        print("Warning: training and testing data do not both contain CBD data!")

                else:
                    if len(train_suburbs) != 0 and len(test_feat_suburbs) != 0:
                        pred_suburbs = train_from_data(method, train_feat_suburb,
                                                       train_label_suburb,
                                                       test_feat_suburbs, save_model, 'suburbs')
                    else:
                        print("Warning: training and testing data do not both contain " +\
                              "suburban/rural data!")

                    if len(train_cbds) != 0 and len(test_feat_cbds) != 0:
                        pred_cbds = train_from_data(method, train_feat_cbd, train_label_cbd,
                                                    test_feat_cbds, save_model, 'CBD')
                    else:
                        print("Warning: training and testing data do not both contain CBD data!")

            endtime = time()
            duration = endtime - starttime
            predict_time.append(duration)

            # Labels are present: print statistics for the height predictions.
            if labels:
                if method == "RFR":
                    if len(pred_suburbs) != 0:
                        ml_funcs.get_statistics(test_labels_suburbs, pred_suburbs, "split",
                                                feature_subset, imp_suburbs)
                        generate_plots.plot_cumulative_errors(test_labels_suburbs, pred_suburbs,
                                                              'suburbs')
                    if len(pred_cbds) != 0:
                        ml_funcs.get_statistics(test_labels_cbds, pred_cbds, "split",
                                                feature_subset, imp_cbds)
                        generate_plots.plot_cumulative_errors(test_labels_cbds, pred_cbds, 'CBDs')
                else:
                    if len(pred_suburbs) != 0:
                        ml_funcs.get_statistics(test_labels_suburbs, pred_suburbs, "split",
                                                feature_subset)
                        generate_plots.plot_cumulative_errors(test_labels_suburbs, pred_suburbs,
                                                              'suburbs')
                    if len(pred_cbds) != 0:
                        ml_funcs.get_statistics(test_labels_cbds, pred_cbds, "split",
                                                feature_subset)
                        generate_plots.plot_cumulative_errors(test_labels_cbds, pred_cbds, 'CBD')

            # Store predictions in database.
            if save_predictions:
                if len(pred_suburbs) != 0:
                    height_values = list(zip(test_suburbs.id, pred_suburbs))
                    db_funcs.store_predictions(cursor, height_values, table, method, 'split')

                if len(pred_cbds) != 0:
                    height_values = list(zip(test_cbds.id, pred_cbds))
                    db_funcs.store_predictions(cursor, height_values, table, method, 'split')

        db_funcs.close_connection(connection, cursor)

        print("\n>> Total duration (s) of reading data " + \
              "into dataframes: {0} ({1})".format(sum(reading_time),
                                                  timedelta(seconds=sum(reading_time))))
        print("\n>> Total duration (s) of the building " + \
              " height predictions: {0} ({1})".format(sum(predict_time),
                                                      timedelta(seconds=sum(predict_time))))

    # No test database is specified, only train the model based on the training data.
    # Useful when training and storing a model to a file.
    else:
        if len(train_suburbs) != 0:
            train_from_data(method, train_feat_suburb, train_label_suburb, np.array([]),
                            save_model, 'suburbs')
        if len(train_cbds) != 0:
            train_from_data(method, train_feat_cbd, train_label_cbd, np.array([]),
                            save_model, 'CBD')


def single_network(train_data, test_db, tables_test, model, scaler, method,
                   save_predictions, labels, save_model, test_subsets, feature_subset):
    """
    Perform the machine learning based on a single training network
    that combines the CBDs and Suburban and Rural areas.
    """

    reading_time = []
    predict_time = []

    if len(train_data) != 0:
        train_features, train_labels = ml_funcs.get_features_and_labels(train_data, "single",
                                                                        test_subsets,
                                                                        feature_subset,
                                                                        labels=True)

    # A database to perform the tests/ predictions on is specified.
    if test_db:
        connection = db_funcs.setup_connection(test_db)
        connection.autocommit = True
        cursor = connection.cursor()

        # If no specific tables are selected, perform predictions
        # for all tables in the specified testing database.
        if not tables_test:
            tables_test = db_funcs.unique_tables(cursor)

        for table in tables_test:
            if table == 'cbds':
                continue

            print(80*'-')
            print(80*'-')

            starttime = time()

            _, _, test_data = db_funcs.read_data(connection, table, training=labels)

            endtime = time()
            duration = endtime - starttime
            reading_time.append(duration)

            if labels:
                test_features, test_labels = ml_funcs.get_features_and_labels(test_data, "single",
                                                                              test_subsets,
                                                                              feature_subset,
                                                                              labels=labels)
            else:
                test_features = ml_funcs.get_features_and_labels(test_data, "single", test_subsets,
                                                                 feature_subset, labels=labels)

            starttime = time()

            if len(train_data) == 0:
                if method == "RFR":
                    predictions, importances = predict_from_model(method, test_features,
                                                                  model, scaler, 'combined')
                else:
                    predictions = predict_from_model(method, test_features, model, scaler,
                                                     'combined')
            else:
                if method == "RFR":
                    predictions, importances = train_from_data(method, train_features,
                                                               train_labels, test_features,
                                                               save_model, 'combined')
                else:
                    predictions = train_from_data(method, train_features, train_labels,
                                                  test_features, save_model, 'combined')

            endtime = time()
            duration = endtime - starttime
            predict_time.append(duration)

            # Labels are present: print statistics for the height predictions.
            if labels:
                if method == "RFR":
                    ml_funcs.get_statistics(test_labels, predictions, "single", feature_subset,
                                            importances)
                else:
                    ml_funcs.get_statistics(test_labels, predictions, "single", feature_subset)
                generate_plots.plot_cumulative_errors(test_labels, predictions, 'combined',)

            # Store predictions in database.
            if save_predictions:
                height_values = list(zip(test_data.id, predictions))
                db_funcs.store_predictions(cursor, height_values, table, method, 'combined')

        db_funcs.close_connection(connection, cursor)

        print("\n>> Total duration (s) of reading data " + \
              "into dataframes: {0} ({1})".format(sum(reading_time),
                                                  timedelta(seconds=sum(reading_time))))
        print("\n>> Total duration (s) of the building " + \
              " height predictions: {0} ({1})".format(sum(predict_time),
                                                      timedelta(seconds=sum(predict_time))))

    # No test database is specified, only train the model based on the training data.
    # Useful when training and storing a model to a file.
    else:
        if len(train_features) != 0:
            train_from_data(method, train_features, train_labels, np.array([]),
                            save_model, 'combined')


def collect_data(database, tables, train=False):
    """
    Based on the provided database and tables,
    retrieve data from the database for different settings:
    CBD, Suburban/Rural and a combination of these two.
    """

    connection = db_funcs.setup_connection(database)
    cursor = connection.cursor()

    # No specific tables were specified, so all tables in the
    # database are used for training.
    if not tables:
        tables = db_funcs.unique_tables(cursor)

    data_suburb, data_cbd, data_full = np.array([]), np.array([]), np.array([])

    # Extract all training data and store it into a pandas DataFrame.
    for i, table in enumerate(tables):
        if i == 0:
            data_suburb, data_cbd, data_full = db_funcs.read_data(connection, table,
                                                                  training=True)
        else:
            suburb, cbd, full = db_funcs.read_data(connection, table, training=train)
            data_suburb = data_suburb.append(suburb)
            data_cbd = data_cbd.append(cbd)
            data_full = data_full.append(full)

    db_funcs.close_connection(connection, cursor)

    return data_full, data_suburb, data_cbd


def file_exists(fname):
    """
    Check for a given filename of the file exists and whether
    it is an actual file.
    """

    if path.exists(fname) and path.isfile(fname):
        return True

    return False


def call_plot_functions(dataframe, env):
    """
    Based on a pandas DataFrame, generate a (flipped)
    violin plot, correlation matrix, F-scores, VIF scores and
    a feature importance boxplot and barplot.
    """

    if not dataframe.empty:
        generate_plots.violin_plots(dataframe, 5, env)
        generate_plots.plot_correlation_matrix(dataframe, env)
        generate_plots.univariate_selection(dataframe, env)
        generate_plots.compute_vif(dataframe, env)
        generate_plots.feature_imp_boxplots(dataframe, env)


def store_plots(train_full, train_suburb, train_cbd, network_type):
    """
    Given the network prediction type, call the plotting
    functions with the correct parameters.
    """

    if network_type == "single":
        call_plot_functions(train_full, 'combined')
    elif network_type == "split":
        call_plot_functions(train_suburb, 'suburbs')
        call_plot_functions(train_cbd, 'CBD')
    else:
        print("Network type is not <single> or <split>, no plots are generated!")


def read_params(fname):
    """
    Read the JSON parameter file.
    Check if the file exists and if we can actually read
    the data.
    """

    if file_exists(fname):
        with open(fname) as filepointer:
            try:
                params = json.load(filepointer)
            except ValueError as error:
                print("Could not load the JSON parameter file:", error)
                sys.exit()
    else:
        print("JSON parameter file does not exist!")
        sys.exit()

    return params


def check_params(network_type, train_db, model_suburbs, model_cbds, model_single_network,
                 scaler_suburbs, scaler_cbds, scaler_single_network):
    """
    Check if certain combinations of parameters are valid,
    based on the network type that is specified.
    """

    if network_type == "split":
        # If no training database, the model files cannot be empty.
        if not train_db and not model_suburbs and not model_cbds and not \
          scaler_suburbs and not scaler_cbds:
            print("Provide model and scaler file for suburban and CBD area morphologies!")
            sys.exit()

        # No training datase, check if the specified model files are valid.
        elif not train_db and (model_suburbs and model_cbds and scaler_suburbs and scaler_cbds):
            if not file_exists(model_suburbs) or not file_exists(model_cbds) \
              or not file_exists(scaler_suburbs) or not file_exists(scaler_cbds):
                print("The model or scaler file for the suburbs or CBDs does not exist!")
                sys.exit()

    elif network_type == "single":
        # If not training database, the model files cannot be empty.
        if not train_db and not model_single_network and not scaler_single_network:
            print("Provide model and scaler file for the single prediction network!")
            sys.exit()

        # No training database, check if the specified model files are valid.
        elif not train_db and (model_single_network and scaler_single_network):
            if not file_exists(model_single_network) or not file_exists(scaler_single_network):
                print("The model or scaler for the single prediction network does not exist!")
                sys.exit()

    else:
        print("The specified network type is not valid! Choose from: <single> or <split>")
        sys.exit()


def main():
    """
    Perform the logic for the height inference problem based on the
    parameters specified in the JSON parameter file.
    """

    parameter_file = "params.json"
    params = read_params(parameter_file)

    train_db = params['db_traindata']
    test_db = params['db_testdata']
    method = params['method']

    # Check if the user specified which databases to use.
    # The training database is not necessary when a saved model is
    # used.
    if len(train_db) == 0 and len(test_db) == 0:
        print("No training and testing database specified!")
        sys.exit()

    # Check if the chosen machine learning method is valid.
    if method not in ("RFR", "SVR", "MLR"):
        print("The selected method is not valid! Choose from: <RFR>, <MLR> or <SVR>")
        sys.exit()

    tables_train = params['tables_traindb']
    tables_test = params['tables_testdb']
    labels = params['labels_for_testdata']

    network_type = params['network_type']
    save_predictions = params['save_predictions']
    save_model = params['save_prediction_model']

    test_subsets = params['test_subsets']
    feature_subset = params['feature_subset']

    # If the testing of subsets is enabled, the list of features
    # cannot be empty.
    if test_subsets and not feature_subset:
        print("No subset of features is specified!")
        sys.exit()

    # Check if the morphology feature is used correctly.
    if "morphology" in feature_subset and network_type == "split":
        print("The morphology feature can only be selected in the single network!")
        sys.exit()

    create_plots = params['generate_plots']

    model_suburbs = params['model_suburbs']
    scaler_suburbs = params['scaler_suburbs']
    model_cbds = params['model_cbds']
    scaler_cbds = params['scaler_cbds']
    model_single_network = params['model_single_network']
    scaler_single_network = params['scaler_single_network']

    # Check if the combinations of parameters in the JSON file is valid.
    check_params(network_type, train_db, model_suburbs, model_cbds, model_single_network,
                 scaler_suburbs, scaler_cbds, scaler_single_network)

    # Extract the training data from the database.
    if train_db:
        train_full, train_suburb, train_cbd = collect_data(train_db, tables_train, True)
    else:
        train_full, train_suburb, train_cbd = [], [], []

    # Generate plots in the Figures folder for the different training sets.
    if create_plots:
        store_plots(train_full, train_suburb, train_cbd, network_type)

    if network_type == "single":
        single_network(train_full, test_db, tables_test, model_single_network,
                       scaler_single_network, method, save_predictions, labels, save_model,
                       test_subsets, feature_subset)
    elif network_type == "split":
        split_network(train_suburb, train_cbd, test_db, tables_test, model_suburbs, scaler_suburbs,
                      model_cbds, scaler_cbds, method, save_predictions, labels, save_model,
                      test_subsets, feature_subset)
    else:
        print("The specified network type is not valid! Choose from: <single> or <split>")
        sys.exit()


if __name__ == '__main__':
    main()
