"""
Database setup and useful functions.
"""

from time import time
import sys
import psycopg2
import psycopg2.extras
import pandas as pd


def setup_connection(dbname):
    """
    Set up the connection to a given database.
    """
    try:
        print("\n>> Setting up a connection to database: {0}".format(dbname))
        return psycopg2.connect(user="postgres",
                                password="postgres",
                                host="127.0.0.1",
                                port="5432",
                                database=dbname)

    except (Exception, psycopg2.Error) as error:
        print("Error while connecting to PostgreSQL;", error)
        sys.exit()


def close_connection(connection, cursor):
    """
    Close the connection to the database including the
    cursor used to perform the queries.
    """
    if cursor:
        cursor.close()
        print(">> Cursor is closed")

    if connection:
        connection.close()
        print(">> PostgreSQL connection is closed")


def unique_tables(cursor):
    """
    Find all tables in a database, excluding the standard tables.
    """

    # Query the current database for all present public tables.
    cursor.execute("SELECT table_name FROM information_schema.tables " +
                   "WHERE table_schema = 'public';")

    # The tables that are also present, but should not be included in the query process.
    skip_tables = set(["geography_columns", "geometry_columns", "spatial_ref_sys",
                       "raster_columns", "raster_overviews"])

    all_tables = set(list(zip(*cursor.fetchall()))[0])

    return all_tables - skip_tables


def read_data(connection, table, extra_features=False, training=False):
    """
    Read data from a database into a Pandas DataFrame.
    """
    print('=== Reading data into GeoPandas DataFrame ===')
    starttime = time()

    print("Reading table: {}".format(table))

    # Test if we want to include non-geometric features.
    if not extra_features:
         # Training data includes height values, non-training data doesn't.
        if training:
            query = "SELECT id, area, compactness, num_neighbours, " + \
                    "num_adjacent_blds, num_vertices, length, width, slimness, rel_height, " + \
                    "complexity, cbd FROM " + table + " WHERE (rel_height >= 3) AND " + \
                    "(rel_height IS NOT NULL);"
        else:
            query = "SELECT id, area, compactness, num_neighbours, " + \
                    "num_adjacent_blds, num_vertices, length, width, slimness, complexity, cbd " + \
                    " FROM " + table + ";"
    else:
        if training:
            query = "SELECT id, area, compactness, num_neighbours, " + \
                    "num_adjacent_blds, num_vertices, length, width, slimness, rel_height, " + \
                    "complexity, cbd, bldg_type, avg_hh_income, avg_hh_size, pop_density, h_mean, " + \
                    "num_amenities FROM " + table + " WHERE (rel_height >= 3) AND (rel_height IS NOT NULL) " +\
                    " AND (h_mean IS NOT NULL) ORDER BY id;"
        else:
            query = "SELECT id, area, compactness, num_neighbours, " + \
                    "num_adjacent_blds, num_vertices, length, width, slimness, complexity, " + \
                    "cbd, bldg_type, avg_hh_income, avg_hh_size, pop_density, h_mean, num_amenities FROM " + \
                    table + " ORDER BY id;"

    data = pd.read_sql_query(query, connection)
    suburb_data = data.loc[data['cbd'] == 0]
    cbd_data = data.loc[data['cbd'] == 1]

    endtime = time()
    duration = endtime - starttime
    print("Time: ", round(duration, 2), "s")
    print(80*'-')

    return suburb_data, cbd_data, data


def store_predictions(cursor, data, table, method, net_type):
    """
    Store the height predictions output by the ML algorithm in the database.
    """

    print(">> Inserting height predictions into the database for the {0} network".format(net_type))

    starttime = time()

    if net_type in ("split", "combined"):
        cursor.execute("ALTER TABLE " + table + " ADD COLUMN IF NOT EXISTS height_" +
                       method + "_" + net_type + " DOUBLE PRECISION;")

        query = "UPDATE " + table + " SET height_" + method + "_" + net_type + \
                " = subquery.pred_height FROM (VALUES %s) AS subquery(id, pred_height) " + \
                " WHERE " + table + ".id = subquery.id;"

    else:
        print("Invalid network type.")
        sys.exit()

    # page_size: the number of values to insert into the database in one query.
    psycopg2.extras.execute_values(cursor, query, data, page_size=int(2e6))

    endtime = time()
    duration = endtime - starttime
    print("Time: ", round(duration, 2), "s")


def store_errors(cursor, data, table, method, net_type):
    """
    Store the relative errors output by the ML algorithm in the database.
    """

    print(">> Inserting the relative errors into the database for the {0} network".format(net_type))

    starttime = time()

    cursor.execute("ALTER TABLE " + table + " ADD COLUMN IF NOT EXISTS rel_error_" +
                   method + " DOUBLE PRECISION;")

    cursor.execute("ALTER TABLE " + table + " ADD COLUMN IF NOT EXISTS perc_error_" +
                   method + " DOUBLE PRECISION;")

    query = "UPDATE " + table + " SET rel_error_" + method + " = subquery.rel_error, " + \
            "perc_error_" + method + " = subquery.perc_error " + \
            "FROM (VALUES %s) AS subquery(id, rel_error, perc_error) WHERE " + \
            table + ".id = subquery.id;"

    # page_size: the number of values to insert into the database in one query.
    psycopg2.extras.execute_values(cursor, query, data, page_size=int(2e6))

    endtime = time()
    duration = endtime - starttime
    print("Time: ", round(duration, 2), "s")
