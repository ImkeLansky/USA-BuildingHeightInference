"""
Generate the results for test areas based on Shapefiles.
"""

import geopandas as gpd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import generate_plots


def emperical_cdf(ground_truth, rfr, svr, mlr, city, env):
    """
    Plot a cumulative error graph showing how the errors are distributed
    over the number of buildings.
    """

    sns.set()
    sns.set_style("white")
    sns.set_style("ticks")

    abs_errors_rf = np.sort(abs(ground_truth - rfr))
    prop_vals_rf = np.linspace(0, 1, len(abs_errors_rf))

    abs_errors_svr = np.sort(abs(ground_truth - svr))
    prop_vals_svr = np.linspace(0, 1, len(abs_errors_svr))

    abs_errors_mlr = np.sort(abs(ground_truth - mlr))
    prop_vals_mlr = np.linspace(0, 1, len(abs_errors_mlr))

    fig, ax = plt.subplots()
    ax.plot(abs_errors_rf, prop_vals_rf, label='RFR')
    ax.plot(abs_errors_svr, prop_vals_svr, label='SVR')
    ax.plot(abs_errors_mlr, prop_vals_mlr, label='MLR')
    ax.set_xlabel("Error [m]")
    ax.set_ylabel("Cumulative Frequency")

    if city == 'Seattle':
        ax.set_xlim([0, 100])
    else:
        ax.set_xlim([0, 8])
    ax.set_ylim([0, 1])
    ax.legend(frameon=False, loc='lower right')

    fig.tight_layout()
    sns.despine()

    if generate_plots.directory_exists("./Figures"):
        plt.savefig("./Figures/Cumulative_Errors_" + city + "_" + env +".pdf",
                    bbox_inches="tight", dpi=300, transparent=True)
    else:
        print("Directory: ./Figures does not exist!")


def print_statistics(predictions, ground_truth):
    """
    Provide statistics for the height predictions based on ground
    truth values.
    """

    mae = mean_absolute_error(ground_truth, predictions)
    print('Mean Absolute Error (MAE):', round(mae, 2))

    mape = np.mean(np.abs((ground_truth - predictions) / ground_truth)) * 100
    print('Mean Absolute Percentage Error (MAPE):', round(mape, 2))

    rmse = np.sqrt(mean_squared_error(ground_truth, predictions))
    print('Root Mean Square Error (RMSE):', round(rmse, 2))

    rmspe = (np.sqrt(np.mean(np.square((ground_truth - predictions) / ground_truth)))) * 100
    print('Root Mean Squared Percentage Error (RMSPE):', round(rmspe, 2))


def stats_split_model():
    """
    Compute the statistics for RFR, SVR and MLR for the test areas.
    """

    print("\n\n>>> Computing statistics for own ML models: SPLIT prediction network <<<")

    # For Astoria and San Diego only the footprints present in also the open city model
    # are included. The other areas contain too many other buildings, so these are just
    # as a whole.
    fnames = ['../Data/Astoria/Buildings/Astoria_2992_Enriched_Final.shp',
              '../Data/Astoria/Buildings/Astoria_2992_Enriched_OnlyUSBlds_Final.shp',
              '../Data/Seattle/Buildings/Seattle_2286_Enriched_Final.shp',
              '../Data/Portland/Buildings/Portland_2992_Enriched_Final.shp']
    cities = ['Astoria(full)', 'Astoria(subset)', 'Seattle', 'Portland']

    for i, filepointer in enumerate(fnames):
        print(80*'-')
        print(">>", cities[i])

        data = gpd.read_file(filepointer)
        data_suburb = data.loc[data['CBD'] == 0]
        data_cbd = data.loc[data['CBD'] == 1]

        if not data_suburb.empty:
            print("\n=== Random Forest Regression (Suburban/ Rural) ===")
            print_statistics(data_suburb['HEIGHT_RF'], data_suburb['pc_rel_hei'])

            print("\n=== Support Vector Regression (Suburban/ Rural) ===")
            print_statistics(data_suburb['HEIGHT_SVR'], data_suburb['pc_rel_hei'])

            print("\n=== Multiple Linear Regression (Suburban/ Rural) ===")
            print_statistics(data_suburb['HEIGHT_MLR'], data_suburb['pc_rel_hei'])

            emperical_cdf(data_suburb['pc_rel_hei'], data_suburb['HEIGHT_RF'],
                          data_suburb['HEIGHT_SVR'], data_suburb['HEIGHT_MLR'],
                          cities[i], 'suburban_split')

        if not data_cbd.empty:
            print("\n=== Random Forest Regression (CBD) ===")
            print_statistics(data_cbd['HEIGHT_RF'], data_cbd['pc_rel_hei'])

            print("\n=== Support Vector Regression (CBD) ===")
            print_statistics(data_cbd['HEIGHT_SVR'], data_cbd['pc_rel_hei'])

            print("\n=== Multiple Linear Regression (CBD) ===")
            print_statistics(data_cbd['HEIGHT_MLR'], data_cbd['pc_rel_hei'])

            emperical_cdf(data_cbd['pc_rel_hei'], data_cbd['HEIGHT_RF'],
                          data_cbd['HEIGHT_SVR'], data_cbd['HEIGHT_MLR'],
                          cities[i], 'CBD_split')


def stats_ocm():
    """
    Compute statistics for the OCM models of the test areas.
    """

    print("\n\n>>> Computing statistics for OCM <<<")

    # For Astoria and San Diego only the footprints present in also the open city model
    # are included. The other areas contain too many other buildings, so these are just
    # as a whole.
    fnames = ['../Data/Astoria/Buildings/OCM/Astoria_OCM_2992_Enriched_Final.shp',
              '../Data/Astoria/Buildings/OCM/Astoria_OCM_2992_Enriched_OnlyUSBlds_Final.shp',
              '../Data/Seattle/Buildings/OCM/Seattle_OCM_2286_Enriched_Final.shp',
              '../Data/Portland/Buildings/OCM/Portland_OCM_2992_Enriched_Final.shp']
    cities = ['Astoria(full)', 'Astoria(subset)', 'Seattle', 'Portland']


    for i, filepointer in enumerate(fnames):
        print(80*'-')
        print(">>", cities[i])

        data = gpd.read_file(filepointer)

        # Only consider building heights >= 3m.
        # Filter the rows based on the selected height percentile.
        height_50 = data.loc[data['height_50'] >= 3]
        height_75 = data.loc[data['height_75'] >= 3]
        height_90 = data.loc[data['pc_rel_hei'] >= 3]

        print("50th HEIGHT PERCENTILE")
        print_statistics(height_50['HEIGHT'], height_50['height_50'])

        print("\n75th HEIGHT PERCENTILE")
        print_statistics(height_75['HEIGHT'], height_75['height_75'])

        print("\n90th HEIGHT PERCENTILE")
        print_statistics(height_90['HEIGHT'], height_90['pc_rel_hei'])


def stats_onemodel():
    """
    Compute the statistics for RFR, SVR and MLR for only one
    prediction model for all buildings (no split CBD and rural/suburban)
    """

    print("\n\n>>> Computing statistics for own ML models: ONE prediction network <<<")

    # For Astoria and San Diego only the footprints present in also the open city model
    # are included. The other areas contain too many other buildings, so these are just
    # as a whole.
    fnames = ['../Data/Astoria/Buildings/OneNetwork_Old/Astoria_2992_onenetwork_Enriched_Final.shp',
              '../Data/Astoria/Buildings/OneNetwork_Old/Astoria_2992_onenetwork_Enriched_OnlyUSBlds_Final.shp',
              '../Data/Seattle/Buildings/OneNetwork_Old/Seattle_2286_onenetwork_Enriched_Final.shp',
              '../Data/Portland/Buildings/OneNetwork_Old/Portland_2992_onenetwork_Enriched_Final.shp']
    cities = ['Astoria(full)', 'Astoria(subset)', 'Seattle', 'Portland']

    for i, filepointer in enumerate(fnames):
        print(80*'-')
        print(">>", cities[i])

        data = gpd.read_file(filepointer)
        data_suburb = data.loc[data['CBD'] == 0]
        data_cbd = data.loc[data['CBD'] == 1]

        if not data_suburb.empty:
            print("\n=== Random Forest Regression (Suburban/ Rural) ===")
            print_statistics(data_suburb['HEIGHT_RF_'], data_suburb['pc_rel_hei'])

            print("\n=== Support Vector Regression (Suburban/ Rural) ===")
            print_statistics(data_suburb['HEIGHT_SVR'], data_suburb['pc_rel_hei'])

            print("\n=== Multiple Linear Regression (Suburban/ Rural) ===")
            print_statistics(data_suburb['HEIGHT_MLR'], data_suburb['pc_rel_hei'])

            emperical_cdf(data_suburb['pc_rel_hei'], data_suburb['HEIGHT_RF_'],
                          data_suburb['HEIGHT_SVR'], data_suburb['HEIGHT_MLR'],
                          cities[i], 'suburban_nomorph')

        if not data_cbd.empty:
            print("\n=== Random Forest Regression (CBD) ===")
            print_statistics(data_cbd['HEIGHT_RF_'], data_cbd['pc_rel_hei'])

            print("\n=== Support Vector Regression (CBD) ===")
            print_statistics(data_cbd['HEIGHT_SVR'], data_cbd['pc_rel_hei'])

            print("\n=== Multiple Linear Regression (CBD) ===")
            print_statistics(data_cbd['HEIGHT_MLR'], data_cbd['pc_rel_hei'])

            emperical_cdf(data_cbd['pc_rel_hei'], data_cbd['HEIGHT_RF_'],
                          data_cbd['HEIGHT_SVR'], data_cbd['HEIGHT_MLR'],
                          cities[i], 'CBD_nomorph')


def stats_onemodel_cbd_feature():
    """
    Compute the statistics for RFR, SVR and MLR for only one
    prediction model for all buildings. The morphology is used
    as an extra feature in the training and predicting process.
    """

    print("\n\n>>> Computing statistics for own ML models: ONE prediction network with Morpoholgy as feature <<<")

    # For Astoria and San Diego only the footprints present in also the open city model
    # are included. The other areas contain too many other buildings, so these are just
    # as a whole.
    fnames = ['../Data/Astoria/Buildings/SingleNetwork/Astoria_SingleNetwork_2992.shp',
              '../Data/Astoria/Buildings/SingleNetwork/Astoria_SingleNetwork_2992_OnlyUSBlds.shp',
              '../Data/Seattle/Buildings/SingleNetwork/Seattle_SingleNetwork_2286.shp',
              '../Data/Portland/Buildings/SingleNetwork/Portland_SingleNetwork_2992.shp']
    cities = ['Astoria(full)', 'Astoria(subset)', 'Seattle', 'Portland']

    for i, filepointer in enumerate(fnames):
        print(80*'-')
        print(">>", cities[i])

        data = gpd.read_file(filepointer)

        print("\n=== Random Forest Regression ===")
        print_statistics(data['H_RFR_COMB'], data['PC_REL_HEI'])

        print("\n=== Support Vector Regression ===")
        print_statistics(data['H_SVR_COMB'], data['PC_REL_HEI'])

        print("\n=== Multiple Linear Regression ===")
        print_statistics(data['H_MLR_COMB'], data['PC_REL_HEI'])

        emperical_cdf(data['PC_REL_HEI'], data['H_RFR_COMB'], data['H_SVR_COMB'],
                      data['H_MLR_COMB'], cities[i], 'combined')


def main():
    """
    Perform all function calls.
    """

    stats_split_model()
    stats_onemodel()
    stats_onemodel_cbd_feature()
    stats_ocm()

if __name__ == '__main__':
    main()
