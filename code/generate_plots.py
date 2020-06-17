"""
Plotting functions.
"""

import string
from os import path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.inspection import permutation_importance
from statsmodels.stats.outliers_influence import variance_inflation_factor
import ml_funcs


def directory_exists(name):
    """
    Check if the directory to store the file exists.
    """

    if path.exists(name) and path.isdir(name):
        return True

    return False


def feature_imp_boxplots(data, env):
    """
    Compute the feature importance based on a random forest.
    1) Impurity-based importance
    2) Permutation importance
    Information:
    https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance_multicollinear.html
    https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance.html
    """

    if env in ('CBD', 'suburbs'):
        net_type = "split"
        feature_names = np.array(['Area', 'Compactness', '#Neighbours', '#Adjacent Buildings',
                                  '#Vertices', 'Length', 'Width', 'Slimness', 'Complexity'])
    elif env == 'combined':
        net_type = "single"
        feature_names = np.array(['Area', 'Compactness', '#Neighbours', '#Adjacent Buildings',
                                  '#Vertices', 'Length', 'Width', 'Slimness', 'Complexity',
                                  'Morphology'])
    else:
        print("Boxplots feature importance: not a valid option.")
        return

    features, labels = ml_funcs.get_features_and_labels(data, net_type, False, [], labels=True)

    train_X, test_X, train_y, test_y = train_test_split(features, labels, test_size=0.2,
                                                        random_state=0)

    train_X_scaled, scaler = ml_funcs.apply_scaling(train_X, 'RF', env)
    test_X_scaled = scaler.transform(test_X)

    if env == 'CBD':
        regressor = RandomForestRegressor(n_estimators=450, min_samples_split=50,
                                          min_samples_leaf=15, max_features='sqrt',
                                          max_depth=14, bootstrap=False, random_state=0,
                                          n_jobs=-1)
    elif env in ('suburbs', 'combined'):
        regressor = RandomForestRegressor(n_estimators=100, min_samples_split=20,
                                          min_samples_leaf=5, max_features='sqrt',
                                          max_depth=None, bootstrap=True, random_state=0,
                                          n_jobs=-1)
    else:
        print("Not a valid environment type")
        return

    regressor.fit(train_X_scaled, train_y)

    fig = plt.figure(figsize=(6, 4))
    sns.set_style("ticks")
    imp = regressor.feature_importances_
    sort_imp = imp.argsort()[::-1]
    barplot = sns.barplot(imp[sort_imp], feature_names[sort_imp], color='steelblue')
    barplot.set_xlabel("Importance")
    fig.tight_layout()
    sns.despine()

    if directory_exists("./Figures"):
        plt.savefig("./Figures/Importances_" + env + ".pdf", bbox_inches="tight", dpi=300,
                    transparent=True)
    else:
        print("Directory: ./Figures does not exist!")

    print("RF train accuracy: %0.3f" % regressor.score(train_X_scaled, train_y))
    print("RF test accuracy: %0.3f" % regressor.score(test_X_scaled, test_y))

    result = permutation_importance(regressor, train_X_scaled, train_y,
                                    n_repeats=25, random_state=0, n_jobs=-1)
    sorted_idx = result.importances_mean.argsort()[::-1]

    sns.set_style("ticks")
    fig, ax = plt.subplots(figsize=(6, 8))
    ax.boxplot(result.importances[sorted_idx].T)
    ax.set_ylabel("Permutation Importance")
    ax.set_xticklabels(labels=feature_names[sorted_idx], rotation=45, horizontalalignment='right')
    fig.tight_layout()
    sns.despine()

    if directory_exists("./Figures"):
        plt.savefig("./Figures/Perm_Importance_" + env + "_Train.pdf", bbox_inches="tight", dpi=300,
                    transparent=True)
    else:
        print("Directory: ./Figures does not exist!")

    result = permutation_importance(regressor, test_X_scaled, test_y,
                                    n_repeats=25, random_state=0, n_jobs=-1)
    sorted_idx = result.importances_mean.argsort()[::-1]

    sns.set_style("ticks")
    fig, ax = plt.subplots(figsize=(6, 8))
    ax.boxplot(result.importances[sorted_idx].T)
    ax.set_ylabel("Permutation Importance")
    ax.set_xticklabels(labels=feature_names[sorted_idx], rotation=45, horizontalalignment='right')
    fig.tight_layout()
    sns.despine()

    if directory_exists("./Figures"):
        plt.savefig("./Figures/Perm_Importance_" + env + "_Test.pdf", bbox_inches="tight", dpi=300,
                    transparent=True)
    else:
        print("Directory: ./Figures does not exist!")


def compute_vif(data, env):
    """
    Compute the variance inflation factor for given data.
    """

    if env == 'CBD':
        X = data[['area', 'num_neighbours', 'width', 'compactness', 'length']]
    elif env == 'suburbs':
        X = data[['area', 'num_adjacent_blds', 'num_neighbours', 'complexity', 'width']]
    elif env == 'combined':
        X = data[['area', 'num_neighbours', 'num_adjacent_blds', 'length', 'cbd']]
    else:
        print("Not a valid environment type")
        return

    X['intercept'] = 1
    vif = pd.DataFrame()
    vif['variables'] = X.columns
    vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    print(vif)


def univariate_selection(data, fname):
    """
    Perform univariate feature selection and compute the F-score of the features.
    """

    sns.set_style("ticks")
    fig = plt.figure(figsize=(6, 4))

    # For the combined training data the area morphology should be considered.
    if fname == 'combined':
        features = np.array(['Area', 'Compactness', '#Neighbours', '#Adjacent Buildings',
                             '#Vertices', 'Length', 'Width', 'Slimness', 'Complexity',
                             'Morphology'])

        feature_data = data[['area', 'compactness', 'num_neighbours', 'num_adjacent_blds',
                             'num_vertices', 'length', 'width', 'slimness', 'complexity',
                             'cbd']]

    else:
        features = np.array(['Area', 'Compactness', '#Neighbours', '#Adjacent Buildings',
                             '#Vertices', 'Length', 'Width', 'Slimness', 'Complexity'])

        feature_data = data[['area', 'compactness', 'num_neighbours', 'num_adjacent_blds',
                             'num_vertices', 'length', 'width', 'slimness', 'complexity']]

    # Select the top 5 features.
    fs = SelectKBest(score_func=f_regression, k=5)
    fs.fit(feature_data, data['rel_height'])
    cols = fs.get_support(indices=True)
    selected_features = feature_data.iloc[:, cols]

    cols = fs.get_support(indices=True)
    selected_features = feature_data.iloc[:, cols]
    print("Top 5 features univariate selection:", selected_features.columns.values)

    norm_scores = fs.scores_ / np.max(fs.scores_)
    sorted_idx = norm_scores.argsort()[::-1]

    barplot = sns.barplot(norm_scores[sorted_idx], features[sorted_idx], color='steelblue')
    barplot.set_xlabel("F-score (normalised)")
    fig.tight_layout()
    sns.despine()

    if directory_exists("./Figures"):
        plt.savefig("./Figures/F_Scores_" + fname + ".pdf", bbox_inches="tight", dpi=300,
                    transparent=True)
    else:
        print("Directory: ./Figures does not exist!")


def plot_correlation_matrix(data, name):
    """
    Plot matrix with the correlations between the different features.
    """

    sns.set_style("ticks")

    # Select all features from the dataframe.
    if name == 'combined':
        features_df = data[['area', 'compactness', 'num_neighbours', 'num_adjacent_blds',
                            'num_vertices', 'length', 'width', 'slimness', 'complexity',
                            'cbd', 'rel_height']]

        features = ['Area', 'Compactness', '#Neighbours', '#Adjacent Buildings',
                    '#Vertices', 'Length', 'Width', 'Slimness', 'Complexity',
                    'Morphology', 'Building Height']
    else:
        features_df = data[['area', 'compactness', 'num_neighbours', 'num_adjacent_blds',
                            'num_vertices', 'length', 'width', 'slimness', 'complexity',
                            'rel_height']]

        features = ['Area', 'Compactness', '#Neighbours', '#Adjacent Buildings',
                    '#Vertices', 'Length', 'Width', 'Slimness', 'Complexity',
                    'Building Height']

    correlation_matrix = features_df.corr()

    fig = plt.figure(figsize=(12, 10))

    # Create mask to only show one halve of the matrix
    mask = np.triu(np.ones_like(correlation_matrix, dtype=np.bool))

    heatmap = sns.heatmap(correlation_matrix,
                          xticklabels=features,
                          yticklabels=features,
                          cmap='RdBu',
                          annot=True,
                          linewidth=0.5, square=True, mask=mask,
                          linewidths=.5,
                          cbar_kws={"shrink": 0.4, "orientation": "horizontal",
                                    "label": "Correlation"},
                          vmin=-1, vmax=1)
    heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=45, horizontalalignment='right')
    heatmap.tick_params(left=False, bottom=False)
    fig.tight_layout()

    if directory_exists("./Figures"):
        plt.savefig("./Figures/Correlation_Matrix_" + name + ".pdf", bbox_inches="tight", dpi=300,
                    transparent=True)
    else:
        print("Directory: ./Figures does not exist!")


def violin_plots(data, num_classes, fname):
    """
    Create violin plots showing the distribution of the data.
    x: features, y: building height
    """

    sns.set(style="ticks", font_scale=0.75)

    # Create figure that contains a subplot for every feature.
    if fname == 'combined':
        _, axs = plt.subplots(10, 1, figsize=(12, 18), sharey=True)
        features = ["area", "compactness", "num_neighbours", "num_adjacent_blds",
                    "num_vertices", "length", "width", "slimness", "complexity",
                    "cbd"]
    else:
        _, axs = plt.subplots(9, 1, figsize=(12, 18), sharey=True)

        features = ["area", "compactness", "num_neighbours", "num_adjacent_blds",
                    "num_vertices", "length", "width", "slimness", "complexity"]

    # For every feature plot the violin bars.
    for num, feature in enumerate(features):

        # Create a temporary dataframe with the two necessary entities.
        tmp = data[[feature, 'rel_height']].copy()

        # Compute the range of the 'bins' based on the range
        # of the data.
        col_name = feature + '_sub'
        range_feature = max(tmp[feature]) - min(tmp[feature])

        # Make sure to use integers for the number of neighbours and
        # adjacent buildings
        if feature in ('num_neighbours', 'num_adjacent_blds'):
            step = range_feature // num_classes
        else:
            step = range_feature / num_classes

        var_order = []

        if feature != 'cbd':
            # Go over every 'bin' and compute its lower and upper boundary.
            # Based on the value of the feature assign a label to each data point.
            for i in range(num_classes):
                if i == 0:
                    thres = min(tmp[feature]) + ((i + 1) * step)
                    name = str(round(min(tmp[feature]), 2)) + ' - ' + str(round(thres, 2))
                    tmp.loc[(tmp[feature] >= min(tmp[feature])) & (tmp[feature] < thres),
                            col_name] = name
                elif i == num_classes - 1:
                    thres = min(tmp[feature]) + (i * step)
                    name = str(round(thres, 2)) + ' - ' + str(round(max(tmp[feature]), 2))
                    tmp.loc[(tmp[feature] >= thres) & (tmp[feature] <= max(tmp[feature])),
                            col_name] = name
                else:
                    thres_1 = min(tmp[feature]) + (i * step)
                    thres_2 = min(tmp[feature]) + ((i + 1) * step)
                    name = str(round(thres_1, 2)) + ' - ' + str(round(thres_2, 2))
                    tmp.loc[(tmp[feature] >= thres_1) & (tmp[feature] < thres_2),
                            col_name] = name

                var_order.append(name)

        # CBD doesn't require splitting into 5 categories, already has 2.
        else:
            tmp.loc[tmp[feature] == 0, col_name] = 'No'
            tmp.loc[tmp[feature] == 1, col_name] = 'Yes'
            var_order = ['Yes', 'No']

        x_name = string.capwords(" ".join(feature.split("_")))

        ax = sns.violinplot(x=col_name, y="rel_height", data=tmp, order=var_order, palette="tab10",
                            ax=axs[num], linewidth=1)
        plt.subplots_adjust(hspace=0.35)
        sns.despine()
        ax.set_xlabel(x_name, fontsize=9)
        ax.set_ylim([-45, 200])
        ax.set_ylabel("Building Height [m]", fontsize=9)

    if directory_exists("./Figures"):
        plt.savefig("./Figures/Violin_" + fname + "_cutoff.pdf", bbox_inches="tight", dpi=300,
                    transparent=True)
    else:
        print("Directory: ./Figures does not exist!")


def violin_plots_flipped(data, num_classes, fname):
    """
    Create flipped violin plots showing the distribution of the data.
    x: building height, y: features
    """

    sns.set(style="ticks", font_scale=0.75)

    # Create figure that contains a subplot for every feature.
    if fname == 'combined':
        _, axs = plt.subplots(10, 1, figsize=(12, 18), sharey=True)
        features = ["area", "compactness", "num_neighbours", "num_adjacent_blds",
                    "num_vertices", "length", "width", "slimness", "complexity",
                    "cbd"]
    else:
        _, axs = plt.subplots(9, 1, figsize=(12, 18), sharey=True)

        features = ["area", "compactness", "num_neighbours", "num_adjacent_blds",
                    "num_vertices", "length", "width", "slimness", "complexity"]

    # The bins for the violin plots (height attribute).
    cbd_thres = [0, 50, 100, 150, 200, max(data['rel_height'])]
    suburb_thres = [0, 10, 20, 50, 100, max(data['rel_height'])]

    # For every feature plot the violin bars.
    for num, feature in enumerate(features):

        # Create a temporary dataframe with the two necessary entities.
        tmp = data[[feature, 'rel_height']].copy()

        col_name = 'rel_height_sub'
        var_order = []

        # Put the heights into bins.
        for i in range(num_classes):
            if fname == 'Violin_CBD':
                name = str(round(cbd_thres[i], 2)) + ' - ' + str(round(cbd_thres[i + 1], 2))
                tmp.loc[(tmp['rel_height'] >= cbd_thres[i]) &
                        (tmp['rel_height'] < cbd_thres[i + 1]), col_name] = name
            elif fname == 'Violin_Non-CBD':
                name = str(round(suburb_thres[i], 2)) + ' - ' + str(round(suburb_thres[i + 1], 2))
                tmp.loc[(tmp['rel_height'] >= suburb_thres[i]) &
                        (tmp['rel_height'] < suburb_thres[i + 1]), col_name] = name
            else:
                print("Not a valid name.")

            var_order.append(name)

        y_name = string.capwords(" ".join(feature.split("_")))

        ax = sns.violinplot(x=col_name, y=feature, data=tmp, order=var_order, palette="tab10",
                            ax=axs[num], linewidth=1)
        plt.subplots_adjust(hspace=0.35)
        sns.despine()
        ax.set_xlabel("Building Height [m]", fontsize=9)
        ax.set_ylabel(y_name, fontsize=9)

        # Change the y-axis of the different plots for better visibility of distributions.
        if feature == 'area':
            ax.set_ylim(-1750, 10000)

        if feature == 'num_adjacent_blds':
            ax.set_ylim(-2.5, 20)

        if feature == 'num_vertices' and fname == 'Violin_Non-CBD':
            ax.set_ylim(-10, 200)

        if feature == 'complexity' and fname == 'Violin_Non-CBD':
            ax.set_ylim(-10, 150)

    if directory_exists("./Figures"):
        plt.savefig("./Figures/" + fname + "_flipped.pdf", bbox_inches="tight", dpi=300,
                    transparent=True)
    else:
        print("Directory: ./Figures does not exist!")


def plot_cumulative_errors(ground_truth, predictions, fname):
    """
    Plot a cumulative error graph showing how the errors are distributed
    over the number of buildings.
    """

    sns.set()
    sns.set_style("white")
    sns.set_style("ticks")

    # https://stackoverflow.com/questions/24788200/calculate-the-cumulative-distribution-function-cdf-in-python
    abs_errors = np.sort(abs(ground_truth - predictions))
    prop_vals = np.linspace(0, 1, len(abs_errors))

    fig, ax = plt.subplots()
    ax.plot(abs_errors, prop_vals)
    ax.set_xlabel("Error [m]")
    ax.set_ylabel("Cumulative Frequency")
    ax.set_title("Cumulative Frequency of Height Errors", fontweight='bold')
    ax.set_xlim([0, 5])
    ax.set_ylim([0, 1])

    fig.tight_layout()

    if directory_exists("./Figures"):
        plt.savefig("./Figures/CumulativeError" + fname + ".pdf", bbox_inches="tight", dpi=300,
                    transparent=True)
    else:
        print("Directory: ./Figures does not exist!")
