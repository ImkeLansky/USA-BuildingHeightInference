"""
For the city of Denver, compare the results for only using geometric features
to a model enriched with also non-geometric features.
"""

from time import time
from math import sqrt
from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import LinearSVR
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import db_funcs
import ml_funcs
import generate_plots


def print_statistics(features, importances, predictions, ground_truth, net_type, method):
    """
    Provide statistics about the height predictions.
    """

    if method == 'RFR':
        print("\nFeature importances {0}:".format(net_type))

        for i, imp in enumerate(importances):
            print(features[i] + ": \t" + str(round(imp, 3)))

    print('\n=== Statistics ===')
    mae = mean_absolute_error(ground_truth, predictions)
    print('Mean Absolute Error (MAE):', round(mae, 2))

    rmse = sqrt(mean_squared_error(ground_truth, predictions))
    print('Root Mean Square Error (RMSE):', round(rmse, 2))

    percentage_error = mean((abs(ground_truth - predictions) / ground_truth) * 100)
    print('Mean Absolute Percentage Error (MAPE):', round(percentage_error, 2))

    rmspe = (np.sqrt(np.mean(np.square((ground_truth - predictions) / ground_truth)))) * 100
    print('Root Mean Squared Percentage Error (RMSPE):', round(rmspe, 2))


def test_geom_features_single(data, cursor, table, store_results, method):
    """
    Only include geometric features during the training and
    prediction process. Based on a single training network.
    """

    features = ["area", "compactness", "num_neighbours", "num_adjacent_blds",
                "num_vertices", "length", "width", "slimness", "complexity",
                "cbd"]
    labels = ["rel_height"]
    dummies = []

    X_train, X_test, y_train, y_test = train_test_split(data[features], data[labels],
                                                        test_size=0.75, random_state=42)

    y_test = y_test.to_numpy().T[0]

    if method == "RFR":
        predictions, importances = randomforest(X_train, y_train,
                                                X_test, features, dummies,
                                                "combined", extra_features=False)

        print_statistics(features, importances, predictions, y_test, "combined", method)
        # generate_plots.plot_cumulative_errors(y_test, predictions, 'combined')


    elif method == "MLR":
        predictions = mlr(X_train, y_train, X_test, features, dummies,
                          "combined", extra_features=False)

        print_statistics(features, None, predictions, y_test, "combined", method)
        # generate_plots.plot_cumulative_errors(y_test, predictions, 'combined')

    elif method == "SVR":
        predictions = svr(X_train, y_train, X_test, features, dummies,
                          "combined", extra_features=False)

        print_statistics(features, None, predictions, y_test, "combined", method)
        # generate_plots.plot_cumulative_errors(y_test, predictions, 'combined')

    else:
        print("Not a valid method.")
        return

    if store_results:
        name = method + "_geometric_single"
        height_vals = list(zip(data.loc[X_test.index].id, predictions))
        db_funcs.store_predictions(cursor, height_vals, table, name, 'combined')

        # Negative: underestimation, positive: overestimation
        # Store relative error and the percentage error in the database.
        rel_errors = (predictions - y_test)
        perc_error = ((predictions - y_test) / y_test) * 100
        error_val = list(zip(data.loc[X_test.index].id, rel_errors, perc_error))
        db_funcs.store_errors(cursor, error_val, table, name, 'combined')


def test_geom_features_split(data_suburb, data_cbd, cursor, table, store_results, method):
    """
    Only include geometric features during the training and
    prediction process. Based on a split training network of suburbs/
    rural areas and CBDs.
    """
    features = ["area", "compactness", "num_neighbours", "num_adjacent_blds",
                "num_vertices", "length", "width", "slimness", "complexity"]
    labels = ["rel_height"]
    dummies = []

    # Split the data into a training and testing set.
    X_sub_train, X_sub_test, y_sub_train, y_sub_test = train_test_split(data_suburb[features],
                                                                        data_suburb[labels],
                                                                        test_size=0.75,
                                                                        random_state=42)

    X_cbd_train, X_cbd_test, y_cbd_train, y_cbd_test = train_test_split(data_cbd[features],
                                                                        data_cbd[labels],
                                                                        test_size=0.75,
                                                                        random_state=42)

    y_sub_test = y_sub_test.to_numpy().T[0]
    y_cbd_test = y_cbd_test.to_numpy().T[0]

    if method == "RFR":
        # Run the random forest regressor for the suburban data and print the results.
        pred_suburbs, imp_suburbs = randomforest(X_sub_train, y_sub_train,
                                                 X_sub_test, features, dummies,
                                                 "suburbs", extra_features=False)

        print_statistics(features, imp_suburbs, pred_suburbs, y_sub_test, "suburbs", method)
        # generate_plots.plot_cumulative_errors(y_sub_test, pred_suburbs, 'suburbs')

        # Run the random forest regressor for the CBD data and print the results.
        pred_cbd, imp_cbd = randomforest(X_cbd_train, y_cbd_train,
                                         X_cbd_test, features, dummies,
                                         "CBD", extra_features=False)

        print_statistics(features, imp_cbd, pred_cbd, y_cbd_test, "CBD", method)
        # generate_plots.plot_cumulative_errors(y_cbd_test, pred_cbd, 'CBD')

    elif method == "MLR":
        # Run the multiple linear regressor for the suburban data and print the results.
        pred_suburbs = mlr(X_sub_train, y_sub_train, X_sub_test,
                           features, dummies, "suburbs", extra_features=False)

        print_statistics(features, None, pred_suburbs, y_sub_test, "suburbs", method)
        # generate_plots.plot_cumulative_errors(y_sub_test, pred_suburbs, 'suburbs')

        # Run the multiple linear regressor for the CBD data and print the results.
        pred_cbd = mlr(X_cbd_train, y_cbd_train, X_cbd_test,
                       features, dummies, "CBD", extra_features=False)

        print_statistics(features, None, pred_cbd, y_cbd_test, "CBD", method)
        # generate_plots.plot_cumulative_errors(y_cbd_test, pred_cbd, 'CBD')

    elif method == "SVR":
        # Run the support vector regressor for the suburban data and print the results.
        pred_suburbs = svr(X_sub_train, y_sub_train, X_sub_test,
                           features, dummies, "suburbs", extra_features=False)

        print_statistics(features, None, pred_suburbs, y_sub_test, "suburbs", method)
        # generate_plots.plot_cumulative_errors(y_sub_test, pred_suburbs, 'suburbs')

        # Run the support vector regressor for the CBD data and print the results.
        pred_cbd = svr(X_cbd_train, y_cbd_train, X_cbd_test,
                       features, dummies, "CBD", extra_features=False)

        print_statistics(features, None, pred_cbd, y_cbd_test, "CBD", method)
        # generate_plots.plot_cumulative_errors(y_cbd_test, pred_cbd, 'CBD')

    else:
        print("Not a valid method.")
        return

    if store_results:
        name = method + "_geometric_split"
        height_vals_suburb = list(zip(data_suburb.loc[X_sub_test.index].id, pred_suburbs))
        db_funcs.store_predictions(cursor, height_vals_suburb, table, name, 'suburbs')

        height_vals_cbd = list(zip(data_cbd.loc[X_cbd_test.index].id, pred_cbd))
        db_funcs.store_predictions(cursor, height_vals_cbd, table, name, 'CBDs')

        # Negative: underestimation, positive: overestimation
        # Store relative error and the percentage error in the database.
        rel_errors_suburbs = (pred_suburbs - y_sub_test)
        perc_error_suburbs = ((pred_suburbs - y_sub_test) / y_sub_test) * 100
        error_val_suburb = list(zip(data_suburb.loc[X_sub_test.index].id, rel_errors_suburbs,
                                    perc_error_suburbs))
        db_funcs.store_errors(cursor, error_val_suburb, table, name, 'suburbs')

        rel_errors_cbd = (pred_cbd - y_cbd_test)
        perc_error_cbd = ((pred_cbd - y_cbd_test) / y_cbd_test) * 100
        error_vals_cbd = list(zip(data_cbd.loc[X_cbd_test.index].id, rel_errors_cbd,
                                  perc_error_cbd))
        db_funcs.store_errors(cursor, error_vals_cbd, table, name, 'CBD')


def test_all_features_single(data, cursor, table, store_results, method):
    """
    Include both geometric and non-geometric features during
    the training and prediction process. Based on a single training network.
    """

    # Create the dummy columns (one hot encoding) for the categorical data.
    cat_columns = ['bldg_type']
    data_processed = pd.get_dummies(data, prefix_sep="__", columns=cat_columns)

    # Extract the names from the dummy columns for later use.
    cat_dummies = [col for col in data_processed if "__" in col \
                   and col.split("__")[0] in cat_columns]

    # Create list of features so we can extract the data from the dataframe.
    features_general = ["area", "compactness", "num_neighbours", "num_adjacent_blds",
                        "num_vertices", "length", "width", "slimness", "complexity",
                        "cbd", "avg_hh_income", "avg_hh_size", "pop_density", "h_mean",
                        "num_amenities"]

    features_all = features_general + cat_dummies
    labels = ["rel_height"]

    X_train, X_test, y_train, y_test = train_test_split(data_processed[features_all],
                                                        data_processed[labels],
                                                        test_size=0.75, random_state=42)

    y_test = y_test.to_numpy().T[0]

    if method == "RFR":
        predictions, importances = randomforest(X_train, y_train,
                                                X_test, features_all, cat_dummies,
                                                "combined", extra_features=False)

        print_statistics(features_all, importances, predictions, y_test, "combined", method)
        # generate_plots.plot_cumulative_errors(y_test, predictions, 'combined')


    elif method == "MLR":
        predictions = mlr(X_train, y_train, X_test, features_all, cat_dummies,
                          "combined", extra_features=False)

        print_statistics(features_all, None, predictions, y_test, "combined", method)
        # generate_plots.plot_cumulative_errors(y_test, predictions, 'combined')

    elif method == "SVR":
        predictions = svr(X_train, y_train, X_test, features_all, cat_dummies,
                          "combined", extra_features=False)

        print_statistics(features_all, None, predictions, y_test, "combined", method)
        # generate_plots.plot_cumulative_errors(y_test, predictions, 'combined')

    else:
        print("Not a valid method.")
        return

    if store_results:
        name = method + "_geometric_single"
        height_vals = list(zip(data.loc[X_test.index].id, predictions))
        db_funcs.store_predictions(cursor, height_vals, table, name, 'combined')

        # Negative: underestimation, positive: overestimation
        # Store relative error and the percentage error in the database.
        rel_errors = (predictions - y_test)
        perc_error = ((predictions - y_test) / y_test) * 100
        error_val = list(zip(data.loc[X_test.index].id, rel_errors, perc_error))
        db_funcs.store_errors(cursor, error_val, table, name, 'combined')


def test_all_features_split(data_suburb, data_cbd, cursor, table, store_results, method):
    """
    Include both geometric and non-geometric features during
    the training and prediction process. Based on a split training network of suburbs/
    rural areas and CBDs.

    Source: https://blog.cambridgespark.com/robust-one-hot-encoding-in-python-3e29bfcec77e
    """

    # Create the dummy columns (one hot encoding) for the categorical data.
    cat_columns = ['bldg_type']
    suburb_processed = pd.get_dummies(data_suburb, prefix_sep="__", columns=cat_columns)
    cbd_processed = pd.get_dummies(data_cbd, prefix_sep="__", columns=cat_columns)

    # Extract the names from the dummy columns for later use.
    cat_dummies_suburb = [col for col in suburb_processed if "__" in col \
                          and col.split("__")[0] in cat_columns]
    cat_dummies_cbd = [col for col in cbd_processed if "__" in col \
                       and col.split("__")[0] in cat_columns]

    # Create list of features so we can extract the data from the dataframe. CBD and suburbs
    # may have separatere categorical features present.
    features_general = ["area", "compactness", "num_neighbours", "num_adjacent_blds",
                        "num_vertices", "length", "width", "slimness", "complexity",
                        "avg_hh_income", "avg_hh_size", "pop_density", "h_mean",
                        "num_amenities"]
    features_suburb = features_general + cat_dummies_suburb
    features_cbd = features_general + cat_dummies_cbd

    labels = ["rel_height"]

    # Split the data into a training and testing set.
    X_sub_train, X_sub_test, y_sub_train, y_sub_test = \
      train_test_split(suburb_processed[features_suburb], suburb_processed[labels],
                       test_size=0.75, random_state=42)

    X_cbd_train, X_cbd_test, y_cbd_train, y_cbd_test = \
      train_test_split(cbd_processed[features_cbd], cbd_processed[labels],
                       test_size=0.75, random_state=42)

    y_sub_test = y_sub_test.to_numpy().T[0]
    y_cbd_test = y_cbd_test.to_numpy().T[0]

    if method == "RFR":
        # Run the random forest regressor for the suburban data and print the results.
        pred_suburbs, imp_suburbs = randomforest(X_sub_train, y_sub_train, X_sub_test,
                                                 features_general, cat_dummies_suburb,
                                                 "suburbs", extra_features=True)

        print_statistics(features_suburb, imp_suburbs, pred_suburbs, y_sub_test, "suburbs", method)
        # generate_plots.plot_cumulative_errors(y_sub_test, pred_suburbs, 'suburbs')

        # Run the random forest regressor for the CBD data and print the results.
        pred_cbd, imp_cbd = randomforest(X_cbd_train, y_cbd_train, X_cbd_test, features_general,
                                         cat_dummies_cbd, "CBD", extra_features=True)

        print_statistics(features_cbd, imp_cbd, pred_cbd, y_cbd_test, "CBD", method)
        # generate_plots.plot_cumulative_errors(y_cbd_test, pred_cbd, 'CBD')

    elif method == "MLR":
        # Run the multiple linear regressor for the suburban data and print the results.
        pred_suburbs = mlr(X_sub_train, y_sub_train, X_sub_test, features_general,
                           cat_dummies_suburb, "suburbs", extra_features=True)

        print_statistics(features_suburb, None, pred_suburbs, y_sub_test, "suburbs", method)
        # generate_plots.plot_cumulative_errors(y_sub_test, pred_suburbs, 'suburbs')

        # Run the multiple linear regressor for the CBD data and print the results.
        pred_cbd = mlr(X_cbd_train, y_cbd_train, X_cbd_test, features_general,
                       cat_dummies_cbd, "CBD", extra_features=True)

        print_statistics(features_cbd, None, pred_cbd, y_cbd_test, "CBD", method)
        # generate_plots.plot_cumulative_errors(y_cbd_test, pred_cbd, 'CBD')

    elif method == "SVR":
        # Run the support vector regressor for the suburban data and print the results.
        pred_suburbs = svr(X_sub_train, y_sub_train, X_sub_test, features_general,
                           cat_dummies_suburb, "suburbs", extra_features=True)

        print_statistics(features_suburb, None, pred_suburbs, y_sub_test, "suburbs", method)
        # generate_plots.plot_cumulative_errors(y_sub_test, pred_suburbs, 'suburbs')

        # Run the support vector regressor for the CBD data and print the results.
        pred_cbd = svr(X_cbd_train, y_cbd_train, X_cbd_test, features_general,
                       cat_dummies_cbd, "CBD", extra_features=True)

        print_statistics(features_cbd, None, pred_cbd, y_cbd_test, "CBD", method)
        # generate_plots.plot_cumulative_errors(y_cbd_test, pred_cbd, 'CBD')

    else:
        print("Not a valid method.")
        return

    if store_results:
        name = method + "_all_split"
        height_vals_suburb = list(zip(data_suburb.loc[X_sub_test.index].id, pred_suburbs))
        db_funcs.store_predictions(cursor, height_vals_suburb, table, name, 'suburbs')

        height_vals_cbd = list(zip(data_cbd.loc[X_cbd_test.index].id, pred_cbd))
        db_funcs.store_predictions(cursor, height_vals_cbd, table, name, 'CBDs')

        # Negative: underestimation, positive: overestimation
        # Store relative error and the percentage error in the database.
        rel_errors_suburbs = (pred_suburbs - y_sub_test)
        perc_error_suburbs = ((pred_suburbs - y_sub_test) / y_sub_test) * 100
        error_val_suburb = list(zip(data_suburb.loc[X_sub_test.index].id, rel_errors_suburbs,
                                    perc_error_suburbs))
        db_funcs.store_errors(cursor, error_val_suburb, table, name, 'suburbs')

        rel_errors_cbd = (pred_cbd - y_cbd_test)
        perc_error_cbd = ((pred_cbd - y_cbd_test) / y_cbd_test) * 100
        error_vals_cbd = list(zip(data_cbd.loc[X_cbd_test.index].id, rel_errors_cbd,
                                  perc_error_cbd))
        db_funcs.store_errors(cursor, error_vals_cbd, table, name, 'CBD')


def randomforest(train_features, train_labels, test_features,
                 names, dummies, net_type, extra_features=False):
    """
    Train the Random Forest Regressor from training data with labels and
    perform predictions on the test data.
    """

    print('\n=== Running Random Forest Regression for {0} ==='.format(net_type))

    regressor = RandomForestRegressor(n_estimators=250, max_features='sqrt',
                                      random_state=0, n_jobs=-1)

    # https://stackoverflow.com/questions/43798377/one-hot-encode-categorical-variables-and-scale-continuous-ones-simultaneouely
    # Only apply feature scaling to the numerical features and not to the one-hot-encoded ones.
    if extra_features:
        train_scaled_tmp, scaler = ml_funcs.apply_scaling(train_features[names], 'RFR',
                                                          net_type, save_scaler=False)
        train_scaled = np.concatenate([train_scaled_tmp, np.array(train_features[dummies])], axis=1)
    else:
        train_scaled, scaler = ml_funcs.apply_scaling(train_features, 'RFR', net_type,
                                                      save_scaler=False)

    # Fit model to the data.
    print('>> Training the network <<')
    starttime = time()
    regressor.fit(train_scaled, train_labels.to_numpy().T[0])
    endtime = time()
    duration_train = endtime - starttime
    print("Time: ", round(duration_train, 2), "s")

    importances = list(regressor.feature_importances_)

    # Make sure to only perform predictions when there are test features.
    # First scale the test features as well.
    if not test_features.empty:

        if extra_features:
            test_scaled_tmp = scaler.transform(test_features[names])
            test_scaled = np.concatenate([test_scaled_tmp, np.array(test_features[dummies])],
                                         axis=1)
        else:
            test_scaled = scaler.transform(test_features)

        print('>> Perform predictions <<')
        starttime = time()
        predictions = regressor.predict(test_scaled)
        endtime = time()
        duration_predict = endtime - starttime
        print("Time: ", round(duration_predict, 2), "s")

        return predictions, importances

    return importances


def mlr(train_features, train_labels, test_features, names, dummies,
        net_type, extra_features=False):
    """
    Train the Multiple Linear Regressor from training data with labels and
    perform predictions on the test data.
    """

    print('\n=== Running Multiple Linear Regression for {0} ==='.format(net_type))

    regressor = LinearRegression(n_jobs=-1)

    if extra_features:
        train_scaled_tmp, scaler = ml_funcs.apply_scaling(train_features[names], 'MLR', net_type,
                                                          save_scaler=False)
        train_scaled = np.concatenate([train_scaled_tmp, np.array(train_features[dummies])], axis=1)
    else:
        train_scaled, scaler = ml_funcs.apply_scaling(train_features, 'MLR', net_type,
                                                      save_scaler=False)

    # Fit model to the data.
    print('>> Training the network <<')
    starttime = time()
    regressor.fit(train_scaled, train_labels.to_numpy().T[0])
    endtime = time()
    duration = endtime - starttime
    print("Time: ", round(duration, 2), "s")

    # Make sure to only perform predictions when there are test features.
    # First scale the test features as well.
    if not test_features.empty:

        if extra_features:
            test_scaled_tmp = scaler.transform(test_features[names])
            test_scaled = np.concatenate([test_scaled_tmp, np.array(test_features[dummies])],
                                         axis=1)
        else:
            test_scaled = scaler.transform(test_features)

        print('>> Perform predictions <<')
        starttime = time()
        predictions = regressor.predict(test_scaled)
        endtime = time()
        duration = endtime - starttime
        print("Time: ", round(duration, 2), "s")

        return predictions


def svr(train_features, train_labels, test_features, names, dummies,
        net_type, extra_features=False):
    """
    Train the Support Vector Regressor from training data with labels and
    perform predictions on the test data.
    """

    print('\n=== Running Support Vector Regression for {0} ==='.format(net_type))

    regressor = LinearSVR(random_state=0, tol=1e-5, max_iter=5000,
                          loss='squared_epsilon_insensitive', epsilon=0.0,
                          C=0.0001, dual=False)

    if extra_features:
        train_scaled_tmp, scaler = ml_funcs.apply_scaling(train_features[names], 'SVR', net_type,
                                                          save_scaler=False)
        train_scaled = np.concatenate([train_scaled_tmp, np.array(train_features[dummies])], axis=1)
    else:
        train_scaled, scaler = ml_funcs.apply_scaling(train_features, 'SVR', net_type,
                                                      save_scaler=False)

    # Fit model to the data.
    print('>> Training the network <<')
    starttime = time()
    regressor.fit(train_scaled, train_labels.to_numpy().T[0])
    endtime = time()
    duration = endtime - starttime
    print("Time: ", round(duration, 2), "s")

    # Make sure to only perform predictions when there are test features.
    # First scale the test features as well.
    if not test_features.empty:

        if extra_features:
            test_scaled_tmp = scaler.transform(test_features[names])
            test_scaled = np.concatenate([test_scaled_tmp, np.array(test_features[dummies])],
                                         axis=1)
        else:
            test_scaled = scaler.transform(test_features)

        print('>> Perform predictions <<')
        starttime = time()
        predictions = regressor.predict(test_scaled)
        endtime = time()
        duration = endtime - starttime
        print("Time: ", round(duration, 2), "s")

        return predictions


def correlation_matrix(data, name):
    """
    Compute the correlation matrix for the non-geometric features
    and the building height.
    """
    sns.set_style("ticks")

    corr_matrix = data[['rel_height', 'avg_hh_income',
                        'avg_hh_size', 'pop_density',
                        'h_mean', 'num_amenities']].corr()

    features = ['Building Height', 'Avg. HH. Income', 'Avg. HH. Size',
                'Population Density', 'Raster Height', '#Amenities']

    fig = plt.figure(figsize=(5, 5))

    # Create mask to only show one halve of the matrix
    mask = np.triu(np.ones_like(corr_matrix, dtype=np.bool))

    heatmap = sns.heatmap(corr_matrix,
                          xticklabels=features,
                          yticklabels=features,
                          cmap='RdBu',
                          annot=True,
                          linewidth=0.5, square=True, mask=mask,
                          linewidths=.5,
                          cbar_kws={"shrink": 0.6, "label": "Correlation"},
                          vmin=-1, vmax=1)
    heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=45, horizontalalignment='right')
    heatmap.tick_params(left=False, bottom=False)
    fig.tight_layout()

    if generate_plots.directory_exists("./Figures"):
        plt.savefig("./Figures/Correlation_NewFeatures_" + name + ".pdf",
                    bbox_inches="tight", dpi=300, transparent=True)
    else:
        print("Directory: ./Figures does not exist!")

    plt.clf()


def violin_plot(data, name):
    """
    Plot violin plots for the building types versus the building height.
    """
    sns.set_style("ticks")

    fig = plt.figure(figsize=(8, 6))

    violin = sns.violinplot(x=data['bldg_type'], y=data['rel_height'], scale='width',
                            width=0.75, color='steelblue')
    violin.set_xticklabels(violin.get_xticklabels(), rotation=45, horizontalalignment='right')

    fig.tight_layout()
    sns.despine()

    violin.set_xlabel('Building Type')
    violin.set_ylabel('Building Height [m]')

    if generate_plots.directory_exists("./Figures"):
        plt.savefig("./Figures/Violin_BldTypes_" + name + ".pdf",
                    bbox_inches="tight", dpi=300, transparent=True)
    else:
        print("Directory: ./Figures does not exist!")


def main():
    """
    Perform all function calls.
    """

    table = "denver_cutout"
    store_results = False
    method = "RFR"

    connection = db_funcs.setup_connection("denver")
    connection.autocommit = True
    cursor = connection.cursor()

    data_suburb, data_cbd, data_full = db_funcs.read_data(connection, table, extra_features=True,
                                                          training=True)

    correlation_matrix(data_suburb, 'suburbs')
    correlation_matrix(data_cbd, 'CBD')
    correlation_matrix(data_full, 'combined')

    violin_plot(data_suburb, 'suburbs')
    violin_plot(data_cbd, 'CBD')
    violin_plot(data_full, 'combined')

    print("\n>>> Running with only geometric features <<<")
    #test_geom_features_split(data_suburb, data_cbd, cursor, table, store_results, method)
    test_geom_features_single(data_full, cursor, table, store_results, method)

    print(80*'-')

    print("\n>>> Running with geometric and non-geometric features <<<")
    #test_all_features_split(data_suburb, data_cbd, cursor, table, store_results, method)
    test_all_features_single(data_full, cursor, table, store_results, method)

    db_funcs.close_connection(connection, cursor)


if __name__ == '__main__':
    main()
