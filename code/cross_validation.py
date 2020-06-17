"""
Perform cross validation for Random Forest Regression and Support
Vector Regression to find the optimal hyperparameters for the models.
"""


import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import LinearSVR
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import db_funcs
import ml_funcs
import generate_plots


def compute_accuracy(predictions, labels):
    """
    Mean Absolute Percent Accuracy
    """

    # Accuracy of training data (mean absolute percentage error)
    errors = abs(predictions - labels)
    mape = 100 * (errors / labels)
    accuracy = 100 - np.mean(mape)

    return accuracy


def rf_n_estimators(train_features, train_labels, test_features, test_labels, name):
    """
    Plot the number of estimators against the accuracy.
    """
    sns.set()
    sns.set_style("ticks")

    train_results = []
    test_results = []

    # The number of trees in the random forest.
    n_estimators = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

    train_scaled, scaler = ml_funcs.apply_scaling(train_features, 'RF', name, save_scaler=False)
    test_scaled = scaler.transform(test_features)

    for estimator in n_estimators:
        print("Num estimators:", estimator)

        randomforest = RandomForestRegressor(n_estimators=estimator, n_jobs=-1, random_state=0)
        randomforest.fit(train_scaled, train_labels)
        predict_train = randomforest.predict(train_scaled)

        # Accuracy of training data (mean absolute percentage error)
        accuracy_train = compute_accuracy(predict_train, train_labels)
        train_results.append(accuracy_train)

        predict_test = randomforest.predict(test_scaled)

        # Accuracy for test data.
        accuracy_test = compute_accuracy(predict_test, test_labels)
        test_results.append(accuracy_test)

    fig = plt.figure(figsize=(10, 6))
    sns.lineplot(x=n_estimators, y=train_results, label='Train')
    sns.lineplot(x=n_estimators, y=test_results, label='Test')
    plt.legend(frameon=False, loc='lower right')
    plt.xlabel('Number of estimators')
    plt.ylabel('Accuracy score [%]')

    fig.tight_layout()
    sns.despine()

    if generate_plots.directory_exists("./Figures"):
        plt.savefig("./Figures/N_Estimators_" + name + ".pdf", bbox_inches="tight", dpi=300,
                    transparent=True)
    else:
        print("Directory: ./Figures does not exist!")


def rf_max_depth(train_features, train_labels, test_features, test_labels, name):
    """
    Plot the maximum tree depth against the accuracy.
    """
    sns.set()
    sns.set_style("ticks")

    train_results = []
    test_results = []

    # Maximum depth of the tree.
    max_depth = np.linspace(1, 35, 35, dtype=int)

    train_scaled, scaler = ml_funcs.apply_scaling(train_features, 'RF', name, save_scaler=False)
    test_scaled = scaler.transform(test_features)

    for depth in max_depth:
        print("Depth:", depth)

        randomforest = RandomForestRegressor(max_depth=depth, n_jobs=-1, random_state=0)
        randomforest.fit(train_scaled, train_labels)
        predict_train = randomforest.predict(train_scaled)

        # Accuracy of training data (mean absolute percentage error)
        accuracy_train = compute_accuracy(predict_train, train_labels)
        train_results.append(accuracy_train)

        predict_test = randomforest.predict(test_scaled)

        # Accuracy for test data.
        accuracy_test = compute_accuracy(predict_test, test_labels)
        test_results.append(accuracy_test)

    fig = plt.figure(figsize=(10, 6))
    sns.lineplot(x=max_depth, y=train_results, label='Train')
    sns.lineplot(x=max_depth, y=test_results, label='Test')
    plt.legend(frameon=False, loc='lower right')
    plt.xlabel('Maximum tree depth')
    plt.ylabel('Accuracy score [%]')

    fig.tight_layout()
    sns.despine()

    if generate_plots.directory_exists("./Figures"):
        plt.savefig("./Figures/Max_Depth_" + name + ".pdf", bbox_inches="tight", dpi=300,
                    transparent=True)
    else:
        print("Directory: ./Figures does not exist!")


def rf_min_samples_split(train_features, train_labels, test_features, test_labels, name):
    """
    Plot the minimum samples to split a node against the accuracy.
    """
    sns.set()
    sns.set_style("ticks")

    train_results = []
    test_results = []

    samples_start = np.linspace(2, 24, 12, dtype=int)
    samples_end = np.linspace(25, 750, num=30, dtype=int)
    min_samples_split = np.hstack((samples_start, samples_end))

    train_scaled, scaler = ml_funcs.apply_scaling(train_features, 'RF', name, save_scaler=False)
    test_scaled = scaler.transform(test_features)

    for samples in min_samples_split:
        print("Samples split:", samples)

        randomforest = RandomForestRegressor(min_samples_split=samples, n_jobs=-1, random_state=0)
        randomforest.fit(train_scaled, train_labels)
        predict_train = randomforest.predict(train_scaled)

        # Accuracy of training data (mean absolute percentage error)
        accuracy_train = compute_accuracy(predict_train, train_labels)
        train_results.append(accuracy_train)

        predict_test = randomforest.predict(test_scaled)

        # Accuracy for test data.
        accuracy_test = compute_accuracy(predict_test, test_labels)
        test_results.append(accuracy_test)

    fig = plt.figure(figsize=(10, 6))
    sns.lineplot(x=min_samples_split, y=train_results, label='Train')
    sns.lineplot(x=min_samples_split, y=test_results, label='Test')
    plt.legend(frameon=False, loc='upper right')
    plt.xlabel('Minimum samples for splitting')
    plt.ylabel('Accuracy score [%]')

    fig.tight_layout()
    sns.despine()

    if generate_plots.directory_exists("./Figures"):
        plt.savefig("./Figures/Min_Samples_Split_" + name + ".pdf", bbox_inches="tight", dpi=300,
                    transparent=True)
    else:
        print("Directory: ./Figures does not exist!")


def rf_min_samples_leaf(train_features, train_labels, test_features, test_labels, name):
    """
    Plot the minimum samples required in a leaf against the accuracy.
    """
    sns.set()
    sns.set_style("ticks")

    train_results = []
    test_results = []

    samples_start = np.linspace(2, 24, 12, dtype=int)
    samples_end = np.linspace(25, 750, num=30, dtype=int)
    min_samples_leaf = np.hstack((samples_start, samples_end))

    train_scaled, scaler = ml_funcs.apply_scaling(train_features, 'RF', name, save_scaler=False)
    test_scaled = scaler.transform(test_features)

    for samples in min_samples_leaf:
        print("Samples leaf:", samples)

        randomforest = RandomForestRegressor(min_samples_leaf=samples, n_jobs=-1, random_state=0)
        randomforest.fit(train_scaled, train_labels)
        predict_train = randomforest.predict(train_scaled)

        # Accuracy of training data (mean absolute percentage error)
        accuracy_train = compute_accuracy(predict_train, train_labels)
        train_results.append(accuracy_train)

        predict_test = randomforest.predict(test_scaled)

        # Accuracy for test data.
        accuracy_test = compute_accuracy(predict_test, test_labels)
        test_results.append(accuracy_test)

    fig = plt.figure(figsize=(10, 6))
    sns.lineplot(x=min_samples_leaf, y=train_results, label='Train')
    sns.lineplot(x=min_samples_leaf, y=test_results, label='Test')
    plt.legend(frameon=False, loc='upper right')
    plt.xlabel('Minimum samples in leaf')
    plt.ylabel('Accuracy score [%]')

    fig.tight_layout()
    sns.despine()

    if generate_plots.directory_exists("./Figures"):
        plt.savefig("./Figures/Min_Samples_Leaf_" + name + ".pdf", bbox_inches="tight", dpi=300,
                    transparent=True)
    else:
        print("Directory: ./Figures does not exist!")


def cv_rf(train_features, train_labels):
    """
    Apply cross validation for the Random Forest Regressor to find its
    optimal hyperparameters based on the training data.
    """

    # The number of trees in the random forest.
    n_estimators = np.linspace(start=50, stop=600, num=12, dtype=int)

    # The number of features to consider at every split of a node.
    max_features = ['auto', 'sqrt', 'log2']

    # The maximum depth of the trees.
    max_depth = [int(x) for x in np.linspace(2, 20, num=10, dtype=int)]
    max_depth.append(None)

    # The minimum number of samples required to split a node.
    min_samples_split = np.linspace(5, 50, num=10, dtype=int)

    # The minimum number of samples required at each leaf node.
    min_samples_leaf = np.linspace(5, 50, num=10, dtype=int)

    # The method for selecting the samples for each individual tree.
    bootstrap = [True, False]

    # Create a random grid with all parameters.
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}

    # Use the random grid to search for best hyperparameters.
    # First create the base model to tune.
    regressor = RandomForestRegressor(random_state=0)

    # Random search of parameters, using 3 fold cross validation,
    # search across 100 different combinations, use 12 processor cores.
    rf_random = RandomizedSearchCV(estimator=regressor,
                                   param_distributions=random_grid,
                                   scoring='neg_mean_absolute_error',
                                   n_iter=75, cv=5, verbose=2,
                                   random_state=0, n_jobs=-1)

    # Scale the features
    train_scaled, _ = ml_funcs.apply_scaling(train_features, 'RF', 'None', save_scaler=False)

    # Fit the random search model.
    search = rf_random.fit(train_scaled, train_labels)

    # Select the parameters that had the best outcome.
    print("RFR best estimator:")
    print(search.best_estimator_)

    print("RFR best hyperparameters of best estimator:")
    print(search.best_estimator_.get_params())

    print("RFR best hyperparameters of search obj:")
    print(search.best_params_)


def svr_epsilon(train_features, train_labels, test_features, test_labels, name):
    """
    Plot epsilon against the accuracy.
    """
    sns.set()
    sns.set_style("ticks")

    train_results = []
    test_results = []

    epsilon = np.linspace(0, 5, 10)

    train_scaled, scaler = ml_funcs.apply_scaling(train_features, 'SVR', name, save_scaler=False)
    test_scaled = scaler.transform(test_features)

    for eps in epsilon:
        print("Epsilon", eps)

        svr = LinearSVR(epsilon=eps, max_iter=2000, random_state=0)
        svr.fit(train_scaled, train_labels)
        predict_train = svr.predict(train_scaled)

        # Accuracy of training data (mean absolute percentage error)
        accuracy_train = compute_accuracy(predict_train, train_labels)
        train_results.append(accuracy_train)

        predict_test = svr.predict(test_scaled)

        # Accuracy for test data.
        accuracy_test = compute_accuracy(predict_test, test_labels)
        test_results.append(accuracy_test)

    fig = plt.figure(figsize=(10, 6))
    sns.lineplot(x=epsilon, y=train_results, label='Train')
    sns.lineplot(x=epsilon, y=test_results, label='Test')
    plt.legend(frameon=False, loc='upper right')
    plt.xlabel('Epsilon')
    plt.ylabel('Accuracy score [%]')

    fig.tight_layout()
    sns.despine()

    if generate_plots.directory_exists("./Figures"):
        plt.savefig("./Figures/Epsilon_" + name + ".pdf", bbox_inches="tight", dpi=300,
                    transparent=True)
    else:
        print("Directory: ./Figures does not exist!")


def svr_C(train_features, train_labels, test_features, test_labels, name):
    """
    Plot C against the accuracy.
    """
    sns.set()
    sns.set_style("ticks")

    train_results = []
    test_results = []

    c_values = np.linspace(1e-4, 1, 10)

    train_scaled, scaler = ml_funcs.apply_scaling(train_features, 'SVR', name, save_scaler=False)
    test_scaled = scaler.transform(test_features)

    for c_val in c_values:
        print("C:", c_val)

        svr = LinearSVR(C=c_val, max_iter=2000, random_state=0)
        svr.fit(train_scaled, train_labels)
        predict_train = svr.predict(train_scaled)

        # Accuracy of training data (mean absolute percentage error)
        accuracy_train = compute_accuracy(predict_train, train_labels)
        train_results.append(accuracy_train)

        predict_test = svr.predict(test_scaled)

        # Accuracy for test data.
        accuracy_test = compute_accuracy(predict_test, test_labels)
        test_results.append(accuracy_test)

    fig = plt.figure(figsize=(10, 6))
    sns.lineplot(x=c_values, y=train_results, label='Train')
    sns.lineplot(x=c_values, y=test_results, label='Test')
    plt.legend(frameon=False, loc='lower right')
    plt.xlabel('C')
    plt.ylabel('Accuracy score [%]')

    fig.tight_layout()
    sns.despine()

    if generate_plots.directory_exists("./Figures"):
        plt.savefig("./Figures/C_" + name + ".pdf", bbox_inches="tight", dpi=300,
                    transparent=True)
    else:
        print("Directory: ./Figures does not exist!")


def svr_maxiter_tolerance(train_features, train_labels, test_features, test_labels, name):
    """
    Plot a combination of the maximum number of iterations and the tolerance
    against the accuracy.
    """
    sns.set()
    sns.set_style("ticks")

    train_results = []
    test_results = []

    tolerances = [1e-3, 1e-4, 1e-5]
    tol_labels = ['1e-3', '1e-4', '1e-5']
    max_iter = np.linspace(100, 5000, 50, dtype=int)

    train_scaled, scaler = ml_funcs.apply_scaling(train_features, 'SVR', name, save_scaler=False)
    test_scaled = scaler.transform(test_features)

    for tolerance in tolerances:
        temp_train = []
        temp_test = []

        print("Tolerance:", tolerance)

        for iteration in max_iter:
            print("Max. iterations:", iteration)

            svr = LinearSVR(tol=tolerance, max_iter=iteration, random_state=0)
            svr.fit(train_scaled, train_labels)
            predict_train = svr.predict(train_scaled)

            # Accuracy of training data (mean absolute percentage error)
            accuracy_train = compute_accuracy(predict_train, train_labels)
            temp_train.append(accuracy_train)

            predict_test = svr.predict(test_scaled)

            # Accuracy for test data.
            accuracy_test = compute_accuracy(predict_test, test_labels)
            temp_test.append(accuracy_test)

        train_results.append(temp_train)
        test_results.append(temp_test)

    fig = plt.figure(figsize=(10, 6))
    for i in range(len(train_results)):
        label_train = 'Train (tol' + tol_labels[i] +')'
        sns.lineplot(x=max_iter, y=train_results[i], label=label_train)
        label_test = 'Test (tol' + tol_labels[i] +')'
        sns.lineplot(x=max_iter, y=test_results[i], label=label_test)

    plt.legend(frameon=False, loc='lower left', bbox_to_anchor=(1.0, 0.0))
    plt.xlabel('Maximum number of iterations')
    plt.ylabel('Accuracy score [%]')

    fig.tight_layout()
    sns.despine()

    if generate_plots.directory_exists("./Figures"):
        plt.savefig("./Figures/MaxIter_Tolerance_" + name + ".pdf", bbox_inches="tight", dpi=300,
                    transparent=True)
    else:
        print("Directory: ./Figures does not exist!")


def cv_svr(train_features, train_labels):
    """
    Perform the k-fold cross validation for the support vector regressor.
    """

    epsilon = [0.0, 0.5, 1.0]
    tol = [1e-3, 1e-4, 1e-5]
    C = [1e-4, 1e-3, 1e-2, 0.1, 1.0]
    loss = ['epsilon_insensitive', 'squared_epsilon_insensitive']
    dual = [True, False]
    max_iter = np.linspace(200, 5000, 25, dtype=int)

    # Create a random grid with all parameters.
    random_grid = {'epsilon': epsilon,
                   'tol': tol,
                   'C': C,
                   'loss': loss,
                   'dual': dual,
                   'max_iter': max_iter}

    # Use the random grid to search for best hyperparameters.
    # First create the base model to tune.
    svr = LinearSVR(random_state=0)

    # Random search of parameters, using 3 fold cross validation,
    # search across 100 different combinations, use 12 processor cores.
    svr_random = RandomizedSearchCV(estimator=svr,
                                    param_distributions=random_grid,
                                    n_iter=75, cv=5, verbose=2,
                                    random_state=0, n_jobs=-1, error_score=0.0)

    # Scale the features
    train_scaled, _ = ml_funcs.apply_scaling(train_features, 'SVR', 'None', save_scaler=False)

    # Fit the random search model.
    search = svr_random.fit(train_scaled, train_labels)

    # Select the parameters that had the best outcome.
    print("SVR best estimator:")
    print(search.best_estimator_)

    print("SVR best hyperparameters of best estimator:")
    print(search.best_estimator_.get_params())

    print("SVR best hyperparameters of search obj:")
    print(search.best_params_)


def test_performance(train_features, train_labels, test_features, test_labels, method, env):
    """
    Test how much impact the hyperparameter tuning had by comparing the
    optimised model to a bare model.
    """

    if method == 'RFR':
        bare = RandomForestRegressor(random_state=0, n_jobs=-1)

        if env == 'CBD':
            optimized = RandomForestRegressor(n_estimators=450, min_samples_split=50,
                                              min_samples_leaf=15, max_features='sqrt',
                                              max_depth=14, bootstrap=False, n_jobs=-1)
        elif env in ('suburbs', 'combined'):
            optimized = RandomForestRegressor(n_estimators=100, min_samples_split=20,
                                              min_samples_leaf=5, max_features='sqrt',
                                              max_depth=None, bootstrap=True, n_jobs=-1)
        else:
            print("Not a valid environment!")
            return None

    elif method == 'SVR':
        bare = LinearSVR(random_state=0)

        if env == 'CBD':
            optimized = LinearSVR(tol=1e-4, max_iter=1800, loss='squared_epsilon_insensitive',
                                  epsilon=1.0, dual=True, C=1e-3)
        elif env == 'suburbs':
            optimized = LinearSVR(tol=1e-5, max_iter=5000, loss='squared_epsilon_insensitive',
                                  epsilon=0.0, dual=False, C=1e-4)
        elif env == 'combined':
            optimized = LinearSVR(random_state=0, tol=0.0001, max_iter=200,
                                  loss='epsilon_insensitive', epsilon=1.0,
                                  C=0.01, dual=True)
        else:
            print("Not a valid environment!")
            return None

    else:
        print("Not a valid method: choose RFR or SVR.")
        return None

    # Scale the features.
    train_scaled, scaler = ml_funcs.apply_scaling(train_features, method, env, save_scaler=False)
    test_scaled = scaler.transform(test_features)

    # Fit the data on the bare model, perform height predictions
    bare.fit(train_scaled, train_labels)
    predictions_bare = bare.predict(test_scaled)
    accuracy_bare = compute_accuracy(predictions_bare, test_labels)
    bare_mae = mean_absolute_error(predictions_bare, test_labels)

    # Now do the same for the optimized model.
    optimized.fit(train_scaled, train_labels)
    predictions_optimized = optimized.predict(test_scaled)
    accuracy_optimized = compute_accuracy(predictions_optimized, test_labels)
    optimized_mae = mean_absolute_error(predictions_optimized, test_labels)

    return accuracy_bare, bare_mae, accuracy_optimized, optimized_mae


def get_data(database, tables):
    """
    Retrieve the data for the suburbs, CBDs and the combined dataset.
    """

    connection = db_funcs.setup_connection(database)
    cursor = connection.cursor()

    data_suburb, data_cbd, data_full = np.array([]), np.array([]), np.array([])

    for i, table in enumerate(tables):
        if i == 0:
            data_suburb, data_cbd, data_full = db_funcs.read_data(connection, table, training=True)
        else:
            suburb, cbd, full = db_funcs.read_data(connection, table, training=True)
            data_suburb = data_suburb.append(suburb)
            data_cbd = data_cbd.append(cbd)
            data_full = data_full.append(full)

    db_funcs.close_connection(connection, cursor)

    return data_full, data_suburb, data_cbd


def main():
    """
    Perform all function calls.
    """

    train_db = "training"
    train_tables = ["toronto", "stgeorge", "wilson", "cedarcity",
                    "junctioncity", "hoodriver", "scio", "nyc"]

    create_plots = False
    check_accuracy = False

    data_full, data_suburb, data_cbd = get_data(train_db, train_tables)

    features_suburb, labels_suburb = ml_funcs.get_features_and_labels(data_suburb, "split",
                                                                      False, [], labels=True)
    features_cbd, labels_cbd = ml_funcs.get_features_and_labels(data_cbd, "split", False, [],
                                                                labels=True)
    features_full, labels_full = ml_funcs.get_features_and_labels(data_full, "single", False,
                                                                 [], labels=True)

    if check_accuracy:
        test_db = "denver"
        test_tables = ["denver_cutout"]

        denver_full, denver_suburb, denver_cbd = get_data(test_db, test_tables)
        denver_feat_suburb, denver_labels_suburb = ml_funcs.get_features_and_labels(denver_suburb,
                                                                                    "split",
                                                                                    False, [],
                                                                                    labels=True)
        denver_feat_cbd, denver_labels_cbd = ml_funcs.get_features_and_labels(denver_cbd, "split",
                                                                              False, [], labels=True)
        denver_feat_full, denver_labels_full = ml_funcs.get_features_and_labels(denver_full, "single",
                                                                                False, [], labels=True)

        # Split up the Denver data to later use it to test the bare model and the improved model.
        # Part of the data is used for training, the other for testing.
        X_train_sub_denv, X_test_sub_denv, y_train_sub_denv, y_test_sub_denv = \
          train_test_split(denver_feat_suburb, denver_labels_suburb, test_size=0.5, random_state=42)
        X_train_cbd_denv, X_test_cbd_denv, y_train_cbd_denv, y_test_cbd_denv = \
          train_test_split(denver_feat_cbd, denver_labels_cbd, test_size=0.5, random_state=42)
        X_train_full_denv, X_test_full_denv, y_train_full_denv, y_test_full_denv = \
          train_test_split(denver_feat_full, denver_labels_full, test_size=0.5, random_state=42)


    ### SINGLE NETWORK ###
    print(">> Cross validation for the entire training dataset")
    cv_svr(features_full, labels_full)
    cv_rf(features_full, labels_full)

    if check_accuracy:
        print(">> Perform check on how much the accuracy improved for Denver CBD")
        acc_bare, bare_mae, acc_opt, opt_mae = test_performance(X_train_full_denv, y_train_full_denv,
                                                                X_test_full_denv, y_test_full_denv,
                                                                'SVR', 'combined')
        print("[SVR] Accuracy bare model CBD:", acc_bare)
        print("[SVR] MAE bare model CBD:", bare_mae)
        print("[SVR] Accuracy optimized model CBD:", acc_opt)
        print("[SVR] MAE optimized model CBD:", opt_mae)
        print(80*'-')

        acc_bare, bare_mae, acc_opt, opt_mae = test_performance(X_train_full_denv, y_train_full_denv,
                                                                X_test_full_denv, y_test_full_denv,
                                                                'RFR', 'combined')
        print("[RFR] Accuracy bare model CBD:", acc_bare)
        print("[RFR] MAE bare model CBD:", bare_mae)
        print("[RFR] Accuracy optimized model CBD:", acc_opt)
        print("[RFR] MAE optimized model CBD:", opt_mae)
        print(80*'-')


    ### CBDS ###
    print("\n>> Cross validation for the CBD data")
    cv_rf(features_cbd, labels_cbd)
    cv_svr(features_cbd, labels_cbd)

    # Use a subset of the CBD data for training and the other for testing to get
    # the overview plots how each hyperparameter influences the accuracy.
    X_train_cbd, X_test_cbd, y_train_cbd, y_test_cbd = train_test_split(features_cbd, labels_cbd,
                                                                        test_size=0.5,
                                                                        random_state=42)

    if create_plots:
        print(">> Generate plots for the CBDs")
        rf_n_estimators(X_train_cbd, y_train_cbd, X_test_cbd, y_test_cbd, 'CBD')
        rf_max_depth(X_train_cbd, y_train_cbd, X_test_cbd, y_test_cbd, 'CBD')
        rf_min_samples_split(X_train_cbd, y_train_cbd, X_test_cbd, y_test_cbd, 'CBD')
        rf_min_samples_leaf(X_train_cbd, y_train_cbd, X_test_cbd, y_test_cbd, 'CBD')
        svr_epsilon(X_train_cbd, y_train_cbd, X_test_cbd, y_test_cbd, 'CBD')
        svr_C(X_train_cbd, y_train_cbd, X_test_cbd, y_test_cbd, 'CBD')
        svr_maxiter_tolerance(X_train_cbd, y_train_cbd, X_test_cbd, y_test_cbd, 'CBD')

    if check_accuracy:
        # Next, check the difference between a bare prediction model and the optimised model.
        print(">> Perform check on how much the accuracy improved for Denver CBD")
        acc_bare, bare_mae, acc_opt, opt_mae = test_performance(X_train_cbd_denv, y_train_cbd_denv,
                                                                X_test_cbd_denv, y_test_cbd_denv,
                                                                'SVR', 'CBD')
        print("[SVR] Accuracy bare model CBD:", acc_bare)
        print("[SVR] MAE bare model CBD:", bare_mae)
        print("[SVR] Accuracy optimized model CBD:", acc_opt)
        print("[SVR] MAE optimized model CBD:", opt_mae)
        print(80*'-')

        acc_bare, bare_mae, acc_opt, opt_mae = test_performance(X_train_cbd_denv, y_train_cbd_denv,
                                                                X_test_cbd_denv, y_test_cbd_denv,
                                                                'RFR', 'CBD')
        print("[RFR] Accuracy bare model CBD:", acc_bare)
        print("[RFR] MAE bare model CBD:", bare_mae)
        print("[RFR] Accuracy optimized model CBD:", acc_opt)
        print("[RFR] MAE optimized model CBD:", opt_mae)
        print(80*'-')


    ### SUBURBS ###
    print("\n>> Cross validation for the rural and suburban data")
    cv_rf(features_suburb, labels_suburb)
    cv_svr(features_suburb, labels_suburb)

    # Use a subset of the suburban and rural data for training and the other for testing to get
    # the overview plots how each hyperparameter influences the accuracy.
    X_train_sub, X_test_sub, y_train_sub, y_test_sub = train_test_split(features_suburb,
                                                                        labels_suburb,
                                                                        test_size=0.5,
                                                                        random_state=42)
    if create_plots:
        print(">> Generate plots for the suburban and rural areas")
        rf_n_estimators(X_train_sub, y_train_sub, X_test_sub, y_test_sub, 'suburbs')
        rf_max_depth(X_train_sub, y_train_sub, X_test_sub, y_test_sub, 'suburbs')
        rf_min_samples_split(X_train_sub, y_train_sub, X_test_sub, y_test_sub, 'suburbs')
        rf_min_samples_leaf(X_train_sub, y_train_sub, X_test_sub, y_test_sub, 'suburbs')
        svr_epsilon(X_train_sub, y_train_sub, X_test_sub, y_test_sub, 'suburbs')
        svr_C(X_train_sub, y_train_sub, X_test_sub, y_test_sub, 'suburbs')
        svr_maxiter_tolerance(X_train_sub, y_train_sub, X_test_sub, y_test_sub, 'suburbs')

    if check_accuracy:
        # Next, check the difference between a bare prediction model and the optimised model.
        print(">> Perform check on how much the accuracy improved for Denver suburbs")
        acc_bare, bare_mae, acc_opt, opt_mae = test_performance(X_train_sub_denv, y_train_sub_denv,
                                                                X_test_sub_denv, y_test_sub_denv,
                                                                'SVR', 'suburbs')
        print("[SVR] Accuracy bare model suburb:", acc_bare)
        print("[SVR] MAE bare model suburb:", bare_mae)
        print("[SVR] Accuracy optimized model suburb:", acc_opt)
        print("[SVR] MAE optimized model suburb:", opt_mae)
        print(80*'-')

        acc_bare, bare_mae, acc_opt, opt_mae = test_performance(X_train_sub_denv, y_train_sub_denv,
                                                                X_test_sub_denv, y_test_sub_denv,
                                                                'RFR', 'suburbs')
        print("[RFR] Accuracy bare model suburb:", acc_bare)
        print("[RFR] MAE bare model suburb:", bare_mae)
        print("[RFR] Accuracy optimized model suburb:", acc_opt)
        print("[RFR] MAE optimized model suburb:", opt_mae)
    print(80*'-')


if __name__ == '__main__':
    main()
