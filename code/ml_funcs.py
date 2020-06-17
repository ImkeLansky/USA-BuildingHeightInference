"""
Machine learning functions useful for the different methods.
"""

from math import sqrt
from time import time, strftime
from statistics import mean
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import LinearSVR
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from joblib import load, dump
import generate_plots


def rf_from_model(features, modelfile, scalerfile, net_type):
    """
    Run the Random Forest Regressor from a pre-trained model and feature scaler.
    """

    print('\n=== Running Random Forest Regression from Model ({0}) ==='.format(net_type))

    print('>> Loading the network <<')
    starttime = time()
    regressor = load(modelfile)
    endtime = time()
    duration = endtime - starttime
    print("Time: ", round(duration, 2), "s")

    # Required on server to set the number of jobs from all processors to
    # a certain amount.
    # regressor.set_params(n_jobs=20)

    print('>> Loading the feature scaler <<')
    starttime = time()
    scaler = load(scalerfile)
    endtime = time()
    duration = endtime - starttime
    print("Time: ", round(duration, 2), "s")

    print('>> Applying feature scaling <<')
    starttime = time()
    features_scaled = scaler.transform(features)
    endtime = time()
    duration = endtime - starttime
    print("Time: ", round(duration, 2), "s")

    print('>> Perform predictions <<')
    starttime = time()
    predictions = regressor.predict(features_scaled)
    endtime = time()
    duration = endtime - starttime
    print("Time: ", round(duration, 2), "s")

    importances = list(regressor.feature_importances_)

    return predictions, importances


def rf_from_traindata(train_features, train_labels, test_features, net_type, save_model=False):
    """
    Train the Random Forest Regressor from training data with labels and
    perform predictions on the test data.
    """

    print('\n=== Running Random Forest Regression from Training Data ({0}) ==='.format(net_type))

    # Choose the correct parameters per area type.
    # !! Tuned on all nine non-geometric features.
    if net_type == 'CBD':
        regressor = RandomForestRegressor(n_estimators=450, min_samples_split=50,
                                          min_samples_leaf=15, max_features='sqrt',
                                          max_depth=14, bootstrap=False, random_state=0, n_jobs=-1)
    elif net_type in ('suburbs', 'combined'):
        regressor = RandomForestRegressor(n_estimators=100, min_samples_split=20,
                                          min_samples_leaf=5, max_features='sqrt',
                                          max_depth=None, bootstrap=True, random_state=0, n_jobs=-1)
    else:
        print("Not a valid environment type")

    train_scaled, scaler = apply_scaling(train_features, 'RFR', net_type, save_scaler=save_model)

    # Fit model to the data.
    print('>> Training the network <<')
    starttime = time()
    regressor.fit(train_scaled, train_labels)
    endtime = time()
    duration_train = endtime - starttime
    print("Time: ", round(duration_train, 2), "s")

    importances = list(regressor.feature_importances_)

    # Saving model for later use so you don't have to retrain the network.
    if save_model:
        timestr = strftime("%H%M%S")
        print('>> Saving the network <<')

        if generate_plots.directory_exists("./Models"):
            dump(regressor, './Models/model_RFR_' + net_type + '_' + timestr + '.sav')
        else:
            print("Directory: ./Models does not exist!")

    # Make sure to only perform predictions when there are test features.
    # First scale the test features as well.
    if len(test_features) != 0:
        test_scaled = scaler.transform(test_features)

        print('>> Perform predictions <<')
        starttime = time()
        predictions = regressor.predict(test_scaled)
        endtime = time()
        duration_predict = endtime - starttime
        print("Time: ", round(duration_predict, 2), "s")

        return predictions, importances, duration_train

    return importances, duration_train


def mlr_from_model(features, modelfile, scalerfile, net_type):
    """
    Run the Multiple Linear Regressor from a pre-trained model and feature scaler.
    """

    print('\n=== Running Multiple Linear Regression from Model ({0}) ==='.format(net_type))

    print('>> Loading the network <<')
    starttime = time()
    regressor = load(modelfile)
    endtime = time()
    duration = endtime - starttime
    print("Time: ", round(duration, 2), "s")

    # Required on server to set the number of jobs from all processors to
    # a certain amount.
    # regressor.set_params(n_jobs=20)

    print('>> Loading the feature scaler <<')
    starttime = time()
    scaler = load(scalerfile)
    endtime = time()
    duration = endtime - starttime
    print("Time: ", round(duration, 2), "s")

    print('>> Applying feature scaling <<')
    starttime = time()
    features_scaled = scaler.transform(features)
    endtime = time()
    duration = endtime - starttime
    print("Time: ", round(duration, 2), "s")

    print('>> Perform predictions <<')
    starttime = time()
    predictions = regressor.predict(features_scaled)
    endtime = time()
    duration = endtime - starttime
    print("Time: ", round(duration, 2), "s")

    return predictions


def mlr_from_traindata(train_features, train_labels, test_features, net_type, save_model=False):
    """
    Train the Multiple Linear Regressor from training data with labels and
    perform predictions on the test data.
    """

    print('\n=== Running Multiple Linear Regression from Training Data ({0}) ==='.format(net_type))

    regressor = LinearRegression(n_jobs=-1)

    train_scaled, scaler = apply_scaling(train_features, 'MLR', net_type, save_scaler=save_model)

    # Fit model to the data.
    print('>> Training the network <<')
    starttime = time()
    regressor.fit(train_scaled, train_labels)
    endtime = time()
    duration = endtime - starttime
    print("Time: ", round(duration, 2), "s")

    # Saving model for later use so you don't have to retrain the network.
    if save_model:
        timestr = strftime("%H%M%S")
        print('>> Saving the network <<')

        if generate_plots.directory_exists("./Models"):
            dump(regressor, './Models/model_MLR_' + net_type + '_' + timestr + '.sav')
        else:
            print("Directory: ./Models does not exist!")

    # Make sure to only perform predictions when there are test features.
    # First scale the test features as well.
    if len(test_features) != 0:
        test_scaled = scaler.transform(test_features)

        print('>> Perform predictions <<')
        starttime = time()
        predictions = regressor.predict(test_scaled)
        endtime = time()
        duration = endtime - starttime
        print("Time: ", round(duration, 2), "s")

        return predictions


def svr_from_model(features, modelfile, scalerfile, net_type):
    """
    Run the Suppor Vector Regressor from a pre-trained model and feature scaler.
    """

    print('\n=== Running Support Vector Regression from Model ({0}) ==='.format(net_type))

    print('>> Loading the network <<')
    starttime = time()
    regressor = load(modelfile)
    endtime = time()
    duration = endtime - starttime
    print("Time: ", round(duration, 2), "s")

    print('>> Loading the feature scaler <<')
    starttime = time()
    scaler = load(scalerfile)
    endtime = time()
    duration = endtime - starttime
    print("Time: ", round(duration, 2), "s")

    print('>> Applying feature scaling <<')
    starttime = time()
    features_scaled = scaler.transform(features)
    endtime = time()
    duration = endtime - starttime
    print("Time: ", round(duration, 2), "s")

    print('>> Perform predictions <<')
    starttime = time()
    predictions = regressor.predict(features_scaled)
    endtime = time()
    duration = endtime - starttime
    print("Time: ", round(duration, 2), "s")

    return predictions


def svr_from_traindata(train_features, train_labels, test_features, net_type, save_model=False):
    """
    Train the Support Vector Regressor from training data with labels and
    perform predictions on the test data.
    """

    print('\n=== Running Support Vector Regression from Training Data ({0}) ==='.format(net_type))

    # Choose the correct parameters per area type.
    # !! Tuned on all nine non-geometric features.
    if net_type == 'CBD':
        regressor = LinearSVR(random_state=0, tol=0.0001, max_iter=1800,
                              loss='squared_epsilon_insensitive', epsilon=1.0,
                              C=0.001, dual=True)
    elif net_type == 'suburbs':
        regressor = LinearSVR(random_state=0, tol=1e-5, max_iter=5000,
                              loss='squared_epsilon_insensitive', epsilon=0.0,
                              C=0.0001, dual=False)
    elif net_type == 'combined':
        regressor = LinearSVR(random_state=0, tol=0.0001, max_iter=200,
                              loss='epsilon_insensitive', epsilon=1.0,
                              C=0.01, dual=True)
    else:
        print("Not a valid environment type")

    train_scaled, scaler = apply_scaling(train_features, 'SVR', net_type, save_scaler=save_model)

    # Fit model to the data.
    print('>> Training the network <<')
    starttime = time()
    regressor.fit(train_scaled, train_labels)
    endtime = time()
    duration = endtime - starttime
    print("Time: ", round(duration, 2), "s")

    # Saving model for later use so you don't have to retrain the network.
    if save_model:
        timestr = strftime("%H%M%S")
        print('>> Saving the network <<')

        if generate_plots.directory_exists("./Models"):
            dump(regressor, './Models/model_SVR_' + net_type + '_' + timestr + '.sav')
        else:
            print("Directory: ./Models does not exist!")

    # Make sure to only perform predictions when there are test features.
    # First scale the test features as well.
    if len(test_features) != 0:
        test_scaled = scaler.transform(test_features)

        print('>> Perform predictions <<')
        starttime = time()
        predictions = regressor.predict(test_scaled)
        endtime = time()
        duration = endtime - starttime
        print("Time: ", round(duration, 2), "s")

        return predictions


def get_features_and_labels(data, network_type, test_subsets, feature_subset, labels=False):
    """
    Extract the feature and label columns from the DataFrame.
    Take into consideration if we are dealing with a single training
    network or a split training network.
    It is also possible to only extract a subset of features.
    """

    if test_subsets:
        data_overview = {"area": data.area, "compactness": data.compactness,
                         "num_neighbours": data.num_neighbours,
                         "num_adjacent_blds": data.num_adjacent_blds,
                         "num_vertices": data.num_vertices, "length": data.length,
                         "width": data.width, "slimness": data.slimness,
                         "complexity": data.complexity, "morphology": data.cbd}

        features = np.array([data_overview[x] for x in feature_subset]).transpose()

    else:
        if network_type == "single":
            features = np.array([data.area, data.compactness, data.num_neighbours,
                                 data.num_adjacent_blds, data.num_vertices, data.length,
                                 data.width, data.slimness, data.complexity, data.cbd]).transpose()
        elif network_type == "split":
            features = np.array([data.area, data.compactness, data.num_neighbours,
                                 data.num_adjacent_blds, data.num_vertices, data.length,
                                 data.width, data.slimness, data.complexity]).transpose()
        else:
            print("Not a valid network type.")

    if labels:
        labels = np.array(data.rel_height)
        return features, labels

    return features


def apply_scaling(train_data, method, net_type, save_scaler=False):
    """
    Apply feature scaling based on a specified training dataset.
    The scaler can be saved to file for later use.
    """

    # Compute the mean and standard deviation to be used for later scaling.
    scaler = StandardScaler().fit(train_data)

    # Perform standardization by centering and scaling.
    train_features_scaled = scaler.transform(train_data)

    # Save the scaler.
    if save_scaler:
        timestr = strftime("%H%M%S")
        print('>> Saving the feature scaler <<')

        if generate_plots.directory_exists("./Models"):
            dump(scaler, './Models/scaler_' + method + '_' + net_type + '_' + timestr + '.sav')
        else:
            print("Directory: ./Models does not exist!")

    return train_features_scaled, scaler


def get_statistics(ground_truth, predictions, network_type, feature_subset, importances=[]):
    """
    Print the Mean Absolute Error (MAE) and Root Mean Square Error (RMSE)
    for the predictions. Also print the importance of each of the features.
    """

    print('\n=== Statistics ===')
    mae = mean_absolute_error(ground_truth, predictions)
    print('Mean Absolute Error (MAE):', round(mae, 2))

    rmse = sqrt(mean_squared_error(ground_truth, predictions))
    print('Root Mean Square Error (RMSE):', round(rmse, 2))

    percentage_error = mean((abs(ground_truth - predictions) / ground_truth) * 100)
    print('Mean Absolute Percentage Error (MAPE):', round(percentage_error, 2))

    rmspe = (np.sqrt(np.mean(np.square((ground_truth - predictions) / ground_truth)))) * 100
    print('Root Mean Squared Percentage Error (RMSPE):', round(rmspe, 2))

    if importances:
        if feature_subset:
            feature_names = feature_subset
        else:
            if network_type == "split":
                feature_names = ["Area", "Compactness", "#Neighbours", "#Adjacent buildings",
                                 "#Vertices", "Length", "Width", "Slimness", "Complexity"]

            elif network_type == "single":
                feature_names = ["Area", "Compactness", "#Neighbours", "#Adjacent buildings",
                                 "#Vertices", "Length", "Width", "Slimness", "Complexity",
                                 "Morphology"]

            else:
                print("Not a valid network type!")

        print("\nFeature importances:")
        for i, imp in enumerate(importances):
            print(feature_names[i] + ": \t" + str(round(imp, 3)))
